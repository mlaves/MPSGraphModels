import Accelerate

public func randomNormal(num: Int,
                         mean: Float,
                         standardDeviation: Float,
                         seed: UInt64? = nil) -> [Float] {
    return [Float](unsafeUninitializedCapacity: num) {
        buffer, unsafeUninitializedCapacity in

        guard var arrayDescriptor = BNNSNDArrayDescriptor(data: buffer, shape: .vector(num)) else {
            fatalError()
        }

        var randomNumberGenerator: BNNSRandomGenerator? = nil
        if let seed = seed {
            randomNumberGenerator = BNNSCreateRandomGeneratorWithSeed(BNNSRandomGeneratorMethodAES_CTR,
                                                                          seed,
                                                                          nil)
        } else {
            randomNumberGenerator = BNNSCreateRandomGenerator(BNNSRandomGeneratorMethodAES_CTR, nil)
        }

        guard let randomNumberGenerator = randomNumberGenerator else {
            fatalError()
        }

        BNNSRandomFillNormalFloat(
            randomNumberGenerator,
            &arrayDescriptor,
            mean,
            standardDeviation)

        unsafeUninitializedCapacity = num
        BNNSDestroyRandomGenerator(randomNumberGenerator)
    }
}

public func randomUniform(num: Int,
                          minimum: Float,
                          maximum: Float,
                          seed: UInt64? = nil) -> [Float] {
    return [Float](unsafeUninitializedCapacity: num) {
        buffer, unsafeUninitializedCapacity in

        guard var arrayDescriptor = BNNSNDArrayDescriptor(data: buffer, shape: .vector(num)) else {
            fatalError()
        }

        var randomNumberGenerator: BNNSRandomGenerator? = nil
        if let seed = seed {
            randomNumberGenerator = BNNSCreateRandomGeneratorWithSeed(BNNSRandomGeneratorMethodAES_CTR,
                                                                          seed,
                                                                          nil)
        } else {
            randomNumberGenerator = BNNSCreateRandomGenerator(BNNSRandomGeneratorMethodAES_CTR, nil)
        }

        guard let randomNumberGenerator = randomNumberGenerator else {
            fatalError()
        }

        BNNSRandomFillUniformFloat(
            randomNumberGenerator,
            &arrayDescriptor,
            minimum,
            maximum)

        unsafeUninitializedCapacity = num
        BNNSDestroyRandomGenerator(randomNumberGenerator)
    }
}

public func initConvWeights(convWeightsShape: [Int], randomSeed: UInt64? = nil) -> [Float] {
    let numel = convWeightsShape.reduce(1, {$0 * $1})
    let in_channels = convWeightsShape[2]
    let kernel_sizes = convWeightsShape[...(convWeightsShape.count-3)]
    let stddev = 1.0 / sqrt(Double(kernel_sizes.reduce(in_channels, {$0 * $1})))
    let convWeightsValues = randomUniform(num: numel,
                                          minimum: Float(-stddev),
                                          maximum: Float(+stddev),
                                          seed: randomSeed)
    return convWeightsValues
}

public func initFCWeights(fcWeightsShape: [Int], randomSeed: UInt64? = nil) -> [Float] {
    let numel = fcWeightsShape.reduce(1, {$0 * $1})
    let in_channels = fcWeightsShape[1]
    let stddev = 1.0 / sqrt(Double(in_channels))
    let fcWeightsValues = randomUniform(num: numel,
                                        minimum: Float(-stddev),
                                        maximum: Float(+stddev),
                                        seed: randomSeed)
    return fcWeightsValues
}
