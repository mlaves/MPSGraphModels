import Foundation
import MPSModels

@_expose(Cxx)
public func test_mps() -> Void
{
    let imageShape: [Int] = [224, 224, 3]
    let imagePixelCount: Int = imageShape.reduce(1, {$0 * $1})
    let imageData = randomNormal(num: Int(imagePixelCount), mean: 0.0, standardDeviation: 1.0, seed: 0)

    let model = VGG16()

    _ = model.forward(inputData: imageData, inputShape: imageShape as [NSNumber])

    let time = ContinuousClock().measure {
        for _ in 0...100 {
            _ = model.forward(inputData: imageData, inputShape: imageShape as [NSNumber])
        }
    }

    print(time)
}

@_expose(Cxx)
public struct VGG16_C {
    let model: VGG16

    public init() {
        model = VGG16()
    }

    public func forward() -> [Float] {
        let imageShape: [Int] = [224, 224, 3]
        let imagePixelCount: Int = imageShape.reduce(1, {$0 * $1})
        let imageData = randomNormal(num: Int(imagePixelCount), mean: 0.0, standardDeviation: 1.0, seed: 0)

        let outputTensor = model.forward(inputData: imageData, inputShape: imageShape as [NSNumber])

        var output: [Float] = [Float](repeating: -1.0, count: Int(truncating: outputTensor.shape[1]))
        outputTensor.mpsndarray().readBytes(&output, strideBytes: nil)

        return output;
    }
}