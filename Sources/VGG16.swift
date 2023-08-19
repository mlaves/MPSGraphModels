import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

public class VGG16 : NSObject {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let graph: MPSGraph
    let graphExe: MPSGraphExecutable
    var inputPlaceholderTensor: MPSGraphTensor
    var labelsPlaceholderTensor: MPSGraphTensor
    var targetTrainingTensors: [MPSGraphTensor]
    var targetInferenceTensors: [MPSGraphTensor]
    var targetTrainingOps: [MPSGraphOperation]
    var targetInferenceOps: [MPSGraphOperation]
    var variableTensors: [MPSGraphTensor]
    let batchSize: Int
    let inChannels: Int
    let inHeight: Int
    let inWidth: Int
    let numClasses: Int
    let batchNorm: Bool
    let learnRate: Double
    let randomSeed: UInt64?

    static func addCBRLayer(graph: MPSGraph,
                            sourceTensor: MPSGraphTensor,
                            desc: MPSGraphConvolution2DOpDescriptor,
                            weightsShape: [Int],
                            weightsValues: UnsafeRawPointer,
                            biasValues: UnsafeRawPointer,
                            batchNormValues: [Float]?,
                            variableTensors: inout [MPSGraphTensor]) -> MPSGraphTensor {
        assert(weightsShape.count == 4)

        // conv op
        let weightsTensor = graph.variable(
            with: Data(bytes: weightsValues,
                       count: weightsShape.reduce(1, {$0 * $1})*MemoryLayout<Float>.stride),
            shape: weightsShape as [NSNumber],
            dataType: .float32,
            name: nil
        )
        let convTensor = graph.convolution2D(sourceTensor,
                                             weights: weightsTensor,
                                             descriptor: desc,
                                             name: nil)

        let biasCount = weightsShape[3]
        let biasTensor = graph.variable(
            with: Data(bytes: biasValues, count: biasCount*MemoryLayout<Float>.stride),
            shape: [biasCount as NSNumber],
            dataType: .float32,
            name: nil
        )
        let convBiasTensor = graph.addition(convTensor,
                                            biasTensor,
                                            name: nil)

        variableTensors += [weightsTensor, biasTensor]

        var batchNormTensor = convBiasTensor
        if let bnValues = batchNormValues {
            // mean and variance are computed per channel over the batch
            // TODO: implement running mean and variance
            let meanTensor = graph.mean(of: convBiasTensor,
                                        axes: [0, 1, 2], name: nil)
            let varianceTensor = graph.variance(of: convBiasTensor,
                                                axes: [0, 1, 2], name: nil)

            // gamma and beta are learnable parameters
            let gammaValues = [Float](repeating: bnValues[2], count: weightsShape[3])
            let batchNormGammaTensor = graph.variable(
                with: Data(bytes: gammaValues, count: weightsShape[3]*MemoryLayout<Float>.stride),
                shape: [1, 1, 1, weightsShape[3] as NSNumber],
                dataType: .float32,
                name: nil
            )
            let betaValues = [Float](repeating: bnValues[3], count: weightsShape[3])
            let batchNormBetaTensor = graph.variable(
                with: Data(bytes: betaValues, count: weightsShape[3]*MemoryLayout<Float>.stride),
                shape: [1, 1, 1, weightsShape[3] as NSNumber],
                dataType: .float32,
                name: nil
            )

            batchNormTensor = graph.normalize(convBiasTensor,
                                              mean: meanTensor,
                                              variance: varianceTensor,
                                              gamma: batchNormGammaTensor,
                                              beta: batchNormBetaTensor,
                                              epsilon: 1e-5,
                                              name: nil)

            variableTensors += [batchNormGammaTensor, batchNormBetaTensor]
        }

        let convActivationTensor = graph.reLU(with: batchNormTensor, name: nil)

        return convActivationTensor
    }

    static func addFullyConnectedLayer(graph: MPSGraph,
                                       sourceTensor: MPSGraphTensor,
                                       weightsShape: [Int],
                                       weightsValues: UnsafeRawPointer,
                                       biasValues: UnsafeRawPointer,
                                       hasActivation: Bool,
                                       variableTensors: inout [MPSGraphTensor]) -> MPSGraphTensor {
        // fc op
        let weightsTensor = graph.variable(
            with: Data(bytes: weightsValues,
                       count: weightsShape.reduce(1, {$0 * $1})*MemoryLayout<Float>.stride),
            shape: weightsShape as [NSNumber],
            dataType: .float32,
            name: nil
        )
        let fcTensor = graph.matrixMultiplication(primary: sourceTensor,
                                                  secondary: weightsTensor,
                                                  name: nil)

        let biasTensor = graph.variable(
            with: Data(bytes: biasValues, count: weightsShape[1]*MemoryLayout<Float>.stride),
            shape: [weightsShape[1] as NSNumber],
            dataType: .float32,
            name: nil
        )
        let fcBiasTensor = graph.addition(fcTensor,
                                          biasTensor,
                                          name: nil)

        variableTensors += [weightsTensor, biasTensor]

        if !hasActivation {
            return fcBiasTensor
        }

        let fcActivationTensor = graph.reLU(with: fcBiasTensor, name: nil)

        return fcActivationTensor
    }

    static func addMaxPool(graph: MPSGraph,
                           sourceTensor: MPSGraphTensor,
                           desc: MPSGraphPooling2DOpDescriptor) -> MPSGraphTensor {
        let maxPoolTensor = graph.maxPooling2D(withSourceTensor: sourceTensor,
                                               descriptor: desc,
                                               name: nil)
        return maxPoolTensor
    }

    static func getAssignOperations(graph: MPSGraph,
                                    lossTensor: MPSGraphTensor,
                                    variableTensors: [MPSGraphTensor],
                                    learnRate: Double) -> [MPSGraphOperation] {
        let gradTensors = graph.gradients(of: lossTensor, with: variableTensors, name: nil)

        let learnRateTensor = graph.constant(learnRate, shape: [1], dataType: .float32)

        var updateOps: [MPSGraphOperation] = []
        for (values, gradients) in gradTensors {
            let updateTensor = graph.stochasticGradientDescent(learningRate: learnRateTensor,
                                                               values: values,
                                                               gradient: gradients,
                                                               name: nil)

            let assign = graph.assign(values, tensor: updateTensor, name: nil)
            updateOps += [assign]
        }

        return updateOps
    }

    let convDesc = MPSGraphConvolution2DOpDescriptor(strideInX: 1,
                                                     strideInY: 1,
                                                     dilationRateInX: 1,
                                                     dilationRateInY: 1,
                                                     groups: 1,
                                                     paddingStyle: .TF_SAME,
                                                     dataLayout: .NHWC,
                                                     weightsLayout: .HWIO)!

    let poolDesc = MPSGraphPooling2DOpDescriptor(kernelWidth: 2,
                                                 kernelHeight: 2,
                                                 strideInX: 2,
                                                 strideInY: 2,
                                                 paddingStyle: .TF_VALID,
                                                 dataLayout: .NHWC)!

    public init (batchSize batch_size: Int = 1,
                 inputChannels in_channels: Int = 3,
                 inputWidth in_width: Int = 224,
                 inputHeight in_height: Int = 224,
                 numClasses num_classes: Int = 1000,
                 batchNorm batch_norm: Bool = true,
                 learnRate learn_rate: Double = 0.01,
                 randomSeed random_seed: UInt64? = nil) {
        device = MTLCreateSystemDefaultDevice()!
        commandQueue = device.makeCommandQueue()!
        graph = MPSGraph()
        batchSize = batch_size
        inChannels = in_channels
        inHeight = in_height
        inWidth = in_width
        numClasses = num_classes
        learnRate = learn_rate
        batchNorm = batch_norm
        randomSeed = random_seed

        inputPlaceholderTensor = graph.placeholder(shape: [batchSize, inHeight, inWidth, inChannels] as [NSNumber],
                                                   dataType: .float32,
                                                   name: "input")
        labelsPlaceholderTensor = graph.placeholder(
            shape: [1, numClasses as NSNumber],
            dataType: .float32,
            name: "labels"
        )
        variableTensors = [MPSGraphTensor]()

        // create data for conv0 weights and biases here
        let conv0WeightsShape: [Int] = [3, 3, inChannels, 64]
        let conv0WeightsValues = initConvWeights(convWeightsShape: conv0WeightsShape, randomSeed: randomSeed)
        let conv0BiasValues = [Float](repeating: 0.0, count: conv0WeightsShape[3])

        var bn0Values: [Float]? = nil
        if batchNorm {
            bn0Values = [0.0, 1.0, 1.0, 0.0] // read from file
        }
        let conv0Tensor = VGG16.addCBRLayer(graph: graph,
                                            sourceTensor: inputPlaceholderTensor,
                                            desc: convDesc,
                                            weightsShape: conv0WeightsShape,
                                            weightsValues: conv0WeightsValues,
                                            biasValues: conv0BiasValues,
                                            batchNormValues: bn0Values,
                                            variableTensors: &variableTensors)

        // create data for conv1 weights and biases here
        let conv1WeightsShape: [Int] = [3, 3, 64, 64]
        let conv1WeightsValues = initConvWeights(convWeightsShape: conv1WeightsShape, randomSeed: randomSeed)
        let conv1BiasValues = [Float](repeating: 0.0, count: conv1WeightsShape[3])

        var bn1Values: [Float]? = nil
        if batchNorm {
            bn1Values = [0.0, 1.0, 1.0, 0.0] // read from file
        }
        let conv1Tensor = VGG16.addCBRLayer(graph: graph,
                                            sourceTensor: conv0Tensor,
                                            desc: convDesc,
                                            weightsShape: conv1WeightsShape,
                                            weightsValues: conv1WeightsValues,
                                            biasValues: conv1BiasValues,
                                            batchNormValues: bn1Values,
                                            variableTensors: &variableTensors)

        // add maxPool0 layer
        let maxPool0Tensor = VGG16.addMaxPool(graph: graph,
                                              sourceTensor: conv1Tensor,
                                              desc: poolDesc)

        // create data for conv2 weights and biases here
        let conv2WeightsShape: [Int] = [3, 3, 64, 128]
        let conv2WeightsValues = initConvWeights(convWeightsShape: conv2WeightsShape, randomSeed: randomSeed)
        let conv2BiasValues = [Float](repeating: 0.0, count: conv2WeightsShape[3])

        var bn2Values: [Float]? = nil
        if batchNorm {
            bn2Values = [0.0, 1.0, 1.0, 0.0] // read from file
        }
        let conv2Tensor = VGG16.addCBRLayer(graph: graph,
                                            sourceTensor: maxPool0Tensor,
                                            desc: convDesc,
                                            weightsShape: conv2WeightsShape,
                                            weightsValues: conv2WeightsValues,
                                            biasValues: conv2BiasValues,
                                            batchNormValues: bn2Values,
                                            variableTensors: &variableTensors)

        // create data for conv3 weights and biases here
        let conv3WeightsShape: [Int] = [3, 3, 128, 128]
        let conv3WeightsValues = initConvWeights(convWeightsShape: conv3WeightsShape, randomSeed: randomSeed)
        let conv3BiasValues = [Float](repeating: 0.0, count: conv3WeightsShape[3])

        var bn3Values: [Float]? = nil
        if batchNorm {
            bn3Values = [0.0, 1.0, 1.0, 0.0] // read from file
        }
        let conv3Tensor = VGG16.addCBRLayer(graph: graph,
                                            sourceTensor: conv2Tensor,
                                            desc: convDesc,
                                            weightsShape: conv3WeightsShape,
                                            weightsValues: conv3WeightsValues,
                                            biasValues: conv3BiasValues,
                                            batchNormValues: bn3Values,
                                            variableTensors: &variableTensors)

        // add maxpool layer
        let maxPool1Tensor = VGG16.addMaxPool(graph: graph,
                                              sourceTensor: conv3Tensor,
                                              desc: poolDesc)

        // create data for conv4 weights and biases here
        let conv4WeightsShape: [Int] = [3, 3, 128, 256]
        let conv4WeightsValues = initConvWeights(convWeightsShape: conv4WeightsShape, randomSeed: randomSeed)
        let conv4BiasValues = [Float](repeating: 0.0, count: conv4WeightsShape[3])

        var bn4Values: [Float]? = nil
        if batchNorm {
            bn4Values = [0.0, 1.0, 1.0, 0.0] // read from file
        }
        let conv4Tensor = VGG16.addCBRLayer(graph: graph,
                                            sourceTensor: maxPool1Tensor,
                                            desc: convDesc,
                                            weightsShape: conv4WeightsShape,
                                            weightsValues: conv4WeightsValues,
                                            biasValues: conv4BiasValues,
                                            batchNormValues: bn4Values,
                                            variableTensors: &variableTensors)

        // create data for conv5 weights and biases here
        let conv5WeightsShape: [Int] = [3, 3, 256, 256]
        let conv5WeightsValues = initConvWeights(convWeightsShape: conv5WeightsShape, randomSeed: randomSeed)
        let conv5BiasValues = [Float](repeating: 0.0, count: conv5WeightsShape[3])

        var bn5Values: [Float]? = nil
        if batchNorm {
            bn5Values = [0.0, 1.0, 1.0, 0.0] // read from file
        }
        let conv5Tensor = VGG16.addCBRLayer(graph: graph,
                                            sourceTensor: conv4Tensor,
                                            desc: convDesc,
                                            weightsShape: conv5WeightsShape,
                                            weightsValues: conv5WeightsValues,
                                            biasValues: conv5BiasValues,
                                            batchNormValues: bn5Values,
                                            variableTensors: &variableTensors)

        // create data for conv6 weights and biases here
        let conv6WeightsShape: [Int] = [3, 3, 256, 256]
        let conv6WeightsValues = initConvWeights(convWeightsShape: conv6WeightsShape, randomSeed: randomSeed)
        let conv6BiasValues = [Float](repeating: 0.0, count: conv6WeightsShape[3])

        var bn6Values: [Float]? = nil
        if batchNorm {
            bn6Values = [0.0, 1.0, 1.0, 0.0] // read from file
        }
        let conv6Tensor = VGG16.addCBRLayer(graph: graph,
                                            sourceTensor: conv5Tensor,
                                            desc: convDesc,
                                            weightsShape: conv6WeightsShape,
                                            weightsValues: conv6WeightsValues,
                                            biasValues: conv6BiasValues,
                                            batchNormValues: bn6Values,
                                            variableTensors: &variableTensors)

        // add maxpool2 layer
        let maxPool2Tensor = VGG16.addMaxPool(graph: graph,
                                              sourceTensor: conv6Tensor,
                                              desc: poolDesc)

        // create data for conv7 weights and biases here
        let conv7WeightsShape: [Int] = [3, 3, 256, 512]
        let conv7WeightsValues = initConvWeights(convWeightsShape: conv7WeightsShape, randomSeed: randomSeed)
        let conv7BiasValues = [Float](repeating: 0.0, count: conv7WeightsShape[3])

        var bn7Values: [Float]? = nil
        if batchNorm {
            bn7Values = [0.0, 1.0, 1.0, 0.0] // read from file
        }
        let conv7Tensor = VGG16.addCBRLayer(graph: graph,
                                            sourceTensor: maxPool2Tensor,
                                            desc: convDesc,
                                            weightsShape: conv7WeightsShape,
                                            weightsValues: conv7WeightsValues,
                                            biasValues: conv7BiasValues,
                                            batchNormValues: bn7Values,
                                            variableTensors: &variableTensors)

        // create data for conv8 weights and biases here
        let conv8WeightsShape: [Int] = [3, 3, 512, 512]
        let conv8WeightsValues = initConvWeights(convWeightsShape: conv8WeightsShape, randomSeed: randomSeed)
        let conv8BiasValues = [Float](repeating: 0.0, count: conv8WeightsShape[3])

        var bn8Values: [Float]? = nil
        if batchNorm {
            bn8Values = [0.0, 1.0, 1.0, 0.0] // read from file
        }
        let conv8Tensor = VGG16.addCBRLayer(graph: graph,
                                            sourceTensor: conv7Tensor,
                                            desc: convDesc,
                                            weightsShape: conv8WeightsShape,
                                            weightsValues: conv8WeightsValues,
                                            biasValues: conv8BiasValues,
                                            batchNormValues: bn8Values,
                                            variableTensors: &variableTensors)

        // create data for conv9 weights and biases here
        let conv9WeightsShape: [Int] = [3, 3, 512, 512]
        let conv9WeightsValues = initConvWeights(convWeightsShape: conv9WeightsShape, randomSeed: randomSeed)
        let conv9BiasValues = [Float](repeating: 0.0, count: conv9WeightsShape[3])

        var bn9Values: [Float]? = nil
        if batchNorm {
            bn9Values = [0.0, 1.0, 1.0, 0.0] // read from file
        }
        let conv9Tensor = VGG16.addCBRLayer(graph: graph,
                                            sourceTensor: conv8Tensor,
                                            desc: convDesc,
                                            weightsShape: conv9WeightsShape,
                                            weightsValues: conv9WeightsValues,
                                            biasValues: conv9BiasValues,
                                            batchNormValues: bn9Values,
                                            variableTensors: &variableTensors)

        // add maxpool3 layer
        let maxPool3Tensor = VGG16.addMaxPool(graph: graph,
                                              sourceTensor: conv9Tensor,
                                              desc: poolDesc)

        // create data for conv10 weights and biases here
        let conv10WeightsShape: [Int] = [3, 3, 512, 512]
        let conv10WeightsValues = initConvWeights(convWeightsShape: conv10WeightsShape, randomSeed: randomSeed)
        let conv10BiasValues = [Float](repeating: 0.0, count: conv10WeightsShape[3])

        var bn10Values: [Float]? = nil
        if batchNorm {
            bn10Values = [0.0, 1.0, 1.0, 0.0] // read from file
        }
        let conv10Tensor = VGG16.addCBRLayer(graph: graph,
                                             sourceTensor: maxPool3Tensor,
                                             desc: convDesc,
                                             weightsShape: conv10WeightsShape,
                                             weightsValues: conv10WeightsValues,
                                             biasValues: conv10BiasValues,
                                             batchNormValues: bn10Values,
                                             variableTensors: &variableTensors)

        // create data for conv11 weights and biases here
        let conv11WeightsShape: [Int] = [3, 3, 512, 512]
        let conv11WeightsValues = initConvWeights(convWeightsShape: conv11WeightsShape, randomSeed: randomSeed)
        let conv11BiasValues = [Float](repeating: 0.0, count: conv11WeightsShape[3])

        var bn11Values: [Float]? = nil
        if batchNorm {
            bn11Values = [0.0, 1.0, 1.0, 0.0] // read from file
        }
        let conv11Tensor = VGG16.addCBRLayer(graph: graph,
                                             sourceTensor: conv10Tensor,
                                             desc: convDesc,
                                             weightsShape: conv11WeightsShape,
                                             weightsValues: conv11WeightsValues,
                                             biasValues: conv11BiasValues,
                                             batchNormValues: bn11Values,
                                             variableTensors: &variableTensors)

        // create data for conv12 weights and biases here
        let conv12WeightsShape: [Int] = [3, 3, 512, 512]
        let conv12WeightsValues = initConvWeights(convWeightsShape: conv12WeightsShape, randomSeed: randomSeed)
        let conv12BiasValues = [Float](repeating: 0.0, count: conv12WeightsShape[3])

        var bn12Values: [Float]? = nil
        if batchNorm {
            bn12Values = [0.0, 1.0, 1.0, 0.0] // read from file
        }
        let conv12Tensor = VGG16.addCBRLayer(graph: graph,
                                             sourceTensor: conv11Tensor,
                                             desc: convDesc,
                                             weightsShape: conv12WeightsShape,
                                             weightsValues: conv12WeightsValues,
                                             biasValues: conv12BiasValues,
                                             batchNormValues: bn12Values,
                                             variableTensors: &variableTensors)

        // add maxpool4 layer
        let maxPool4Tensor = VGG16.addMaxPool(graph: graph,
                                              sourceTensor: conv12Tensor,
                                              desc: poolDesc)

        // flatten the output
        let flattenedTensor = graph.reshape(maxPool4Tensor,
                                            shape: [-1, 7*7*512] as [NSNumber],
                                            name: nil)

        // create data for fc0 weights and biases here
        let fc0WeightsShape: [Int] = [25088, 4096]
        let fc0WeightsValues = initFCWeights(fcWeightsShape: fc0WeightsShape, randomSeed: randomSeed)
        let fc0BiasValues = [Float](repeating: 0.0, count: fc0WeightsShape[0])
        let fc0Tensor = VGG16.addFullyConnectedLayer(graph: graph,
                                                     sourceTensor: flattenedTensor,
                                                     weightsShape: fc0WeightsShape,
                                                     weightsValues: fc0WeightsValues,
                                                     biasValues: fc0BiasValues,
                                                     hasActivation: true,
                                                     variableTensors: &variableTensors)

        // let fc0Tensor_drop = graph.dropout(fc0Tensor, rate: 0.2, name: nil)

        // create data for fc1 weights and biases here
        let fc1WeightsShape: [Int] = [4096, 4096]
        let fc1WeightsValues = initFCWeights(fcWeightsShape: fc1WeightsShape, randomSeed: randomSeed)
        let fc1BiasValues = [Float](repeating: 0.0, count: fc1WeightsShape[0])
        let fc1Tensor = VGG16.addFullyConnectedLayer(graph: graph,
                                                     sourceTensor: fc0Tensor,
                                                     weightsShape: fc1WeightsShape,
                                                     weightsValues: fc1WeightsValues,
                                                     biasValues: fc1BiasValues,
                                                     hasActivation: true,
                                                     variableTensors: &variableTensors)


        // let fc1Tensor_drop = graph.dropout(fc1Tensor, rate: 0.2, name: nil)

        // create data for fc2 weights and biases here
        let fc2WeightsShape: [Int] = [4096, numClasses]
        let fc2WeightsValues = initFCWeights(fcWeightsShape: fc2WeightsShape, randomSeed: randomSeed)
        let fc2BiasValues = [Float](repeating: 0.0, count: fc2WeightsShape[0])
        let fc2Tensor = VGG16.addFullyConnectedLayer(graph: graph,
                                                     sourceTensor: fc1Tensor,
                                                     weightsShape: fc2WeightsShape,
                                                     weightsValues: fc2WeightsValues,
                                                     biasValues: fc2BiasValues,
                                                     hasActivation: false,
                                                     variableTensors: &variableTensors)

        // add final softmax layer
        let softmaxTensor = graph.softMax(with: fc2Tensor, axis: -1, name: nil)

        // add loss layer for training
        let lossTensor = graph.softMaxCrossEntropy(fc2Tensor,
                                                   labels: labelsPlaceholderTensor,
                                                   axis: -1,
                                                   reuctionType: .sum,
                                                   name: nil)

        let batchSizeTensor = graph.constant(Double(batchSize),
                                             shape: [1],
                                             dataType: .float32)
        let lossMeanTensor = graph.division(lossTensor, batchSizeTensor, name: nil)

        targetInferenceTensors = [softmaxTensor]
        targetInferenceOps = []

        targetTrainingTensors = [lossMeanTensor, fc2Tensor]
        targetTrainingOps = VGG16.getAssignOperations(graph: graph,
                                                      lossTensor: lossMeanTensor,
                                                      variableTensors: variableTensors,
                                                      learnRate: learnRate)

        graphExe = graph.compile(
            with: nil,
            feeds: [inputPlaceholderTensor : MPSGraphShapedType(
                shape: [batchSize, inHeight, inWidth, inChannels] as [NSNumber],
                dataType: .float32
            )],
            targetTensors: targetInferenceTensors,
            targetOperations: nil,
            compilationDescriptor: nil
        )

        super.init()
    }

    public func trainBatch(inputTensorData: MPSGraphTensorData,
                           labelsTensorData: MPSGraphTensorData) -> MPSGraphTensorData {
        let feeds = [inputPlaceholderTensor: inputTensorData,
                     labelsPlaceholderTensor: labelsTensorData]
        let results = graph.run(with: commandQueue,
                                feeds: feeds,
                                targetTensors: targetTrainingTensors,
                                targetOperations: targetTrainingOps)

        return (results[targetTrainingTensors[1]])!
    }

    private func forwardExe(inputImage: MPSImage) -> MPSGraphTensorData {
        var results: [MPSGraphTensorData]? = nil
        autoreleasepool {
            results = graphExe.run(with: commandQueue,
                                   inputs: [MPSGraphTensorData([inputImage])],
                                   results: nil,
                                   executionDescriptor: nil)
        }

        return (results?[0])!
    }

    public func forward(inputImage: MPSImage) -> MPSGraphTensorData {
        let inputImageTensorData = MPSGraphTensorData([inputImage])

        var results: [MPSGraphTensor: MPSGraphTensorData]? = nil
        autoreleasepool {
            results = graph.run(with: commandQueue,
                                feeds: [inputPlaceholderTensor: inputImageTensorData],
                                targetTensors: targetInferenceTensors,
                                targetOperations: nil)
        }

        return (results?[targetInferenceTensors[0]])!
    }

    public func forward(inputData: UnsafeRawPointer, inputShape: [NSNumber]) -> MPSGraphTensorData {
        let inputImage = MPSImage(
            device: device,
            imageDescriptor: MPSImageDescriptor(channelFormat: .unorm8,
                                                width: inputShape[1].intValue,
                                                height: inputShape[0].intValue,
                                                featureChannels: inputShape[2].intValue)
        )
        inputImage.writeBytes(inputData, dataLayout: .HeightxWidthxFeatureChannels, imageIndex: 0)
        return forward(inputImage: inputImage)
        // return forwardExe(inputImage: inputImage)
    }
}
