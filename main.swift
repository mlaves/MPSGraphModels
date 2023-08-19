import Foundation
import MPSModels
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

func main() {
    let imageShape: [Int] = [224, 224, 1]
    let imagePixelCount: Int = imageShape.reduce(1, {$0 * $1})
    let imageData = randomNormal(num: Int(imagePixelCount), mean: 0.0, standardDeviation: 1.0, seed: 0)

    let model = VGG16(batchSize: 1,
                      inputChannels: imageShape[2],
                      inputWidth: imageShape[0],
                      inputHeight: imageShape[1],
                      numClasses: 10,
                      batchNorm: true)

    var outputTensor = model.forward(inputData: imageData, inputShape: imageShape as [NSNumber])

    let time = ContinuousClock().measure {
        for _ in 0...1000 {
            outputTensor = model.forward(inputData: imageData, inputShape: imageShape as [NSNumber])
        }
    }

    print("Elapsed:", time)

    var output: [Float] = [Float](repeating: 0.0, count: Int(1000))
    outputTensor.mpsndarray().readBytes(&output, strideBytes: nil)
    print(output[0])
    print("fin")
}


func main_train() {
    let imageShape: [Int] = [1, 224, 224, 3]
    let imagePixelCount: Int = imageShape.reduce(1, {$0 * $1})
    var imageData = randomNormal(num: Int(imagePixelCount), mean: 0.0, standardDeviation: 1.0, seed: 0)
    var targetData: [Float] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    let model = VGG16(batchSize: 1,
                      inputChannels: imageShape[3],
                      inputWidth: imageShape[1],
                      inputHeight: imageShape[2],
                      numClasses: 10,
                      batchNorm: true,
                      learnRate: 0.001,
                      randomSeed: 0)

    let inputTensorDataDesc = MPSNDArrayDescriptor(dataType: .float32,
                                                   shape: [1, 224, 224, 3] as [NSNumber])
    let inputTensorData = MPSNDArray(device: MTLCreateSystemDefaultDevice()!,
                                     descriptor: inputTensorDataDesc)
    inputTensorData.writeBytes(&imageData, strideBytes: nil)
    let labelsTensorDataDesc = MPSNDArrayDescriptor(dataType: .float32,
                                                   shape: [1, 10] as [NSNumber])
    let labelsTensorData = MPSNDArray(device: MTLCreateSystemDefaultDevice()!,
                                     descriptor: labelsTensorDataDesc)
    labelsTensorData.writeBytes(&targetData, strideBytes: nil)

    for _ in 0...3 {
        let lossTensorData = model.trainBatch(
            inputTensorData: MPSGraphTensorData(inputTensorData),
            labelsTensorData: MPSGraphTensorData(labelsTensorData)
        )

        let count: Int = lossTensorData.shape.map({$0.intValue}).reduce(1, {$0 * $1})
        var loss: [Float] = [Float](repeating: -1, count: count)
        lossTensorData.mpsndarray().readBytes(&loss, strideBytes: nil)
        print(loss)
    }
}


func main2() {
    // read weights from JSON as base64 string and decode
    let data = try! Data(contentsOf: URL(fileURLWithPath: "../vgg16.json"), options: .mappedIfSafe)
    let jsonResult = try! JSONSerialization.jsonObject(with: data, options: .mutableLeaves)
    if let jsonResult = jsonResult as? Dictionary<String, String> {
        if let base64Data = Data(base64Encoded: jsonResult["conv0weights"]!) {
            var arr = [Float](repeating: 0, count: base64Data.count/MemoryLayout<Float32>.stride)
            _ = arr.withUnsafeMutableBytes { base64Data.copyBytes(to: $0) }
            print(arr)
        }
    }
}

main()
