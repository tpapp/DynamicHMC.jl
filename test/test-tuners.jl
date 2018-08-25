@testset "tuner framework" begin
    s = StepsizeTuner(10)
    @test length(s) == 10
    @test repr(s) == "Stepsize tuner, 10 samples"
    c = StepsizeCovTuner(19, 7.0)
    @test length(c) == 19
    @test repr(c) ==
        "Stepsize and covariance tuner, 19 samples, regularization 7.0"
    b = bracketed_doubling_tuner() # testing the defaults
    @test b isa TunerSequence
    @test b.tuners == (StepsizeTuner(75), # init
                       StepsizeCovTuner(25, 5.0), # doubling each step
                       StepsizeCovTuner(50, 5.0),
                       StepsizeCovTuner(100, 5.0),
                       StepsizeCovTuner(200, 5.0),
                       StepsizeCovTuner(400, 5.0),
                       StepsizeTuner(50)) # terminate
    @test repr(b) ==
        """
Sequence of 7 tuners, 900 total samples
  Stepsize tuner, 75 samples
  Stepsize and covariance tuner, 25 samples, regularization 5.0
  Stepsize and covariance tuner, 50 samples, regularization 5.0
  Stepsize and covariance tuner, 100 samples, regularization 5.0
  Stepsize and covariance tuner, 200 samples, regularization 5.0
  Stepsize and covariance tuner, 400 samples, regularization 5.0
  Stepsize tuner, 50 samples"""
end
