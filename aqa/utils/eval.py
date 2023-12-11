def eval(config):
    # to avoid circular imports
    from aqa.benchmarks import build_benchmark
    from aqa.models import build_model

    benchmark = build_benchmark(config)
    model = build_model(config)

    times = config.EVAL.TIMES
    num_exp = config.EVAL.NUM_EXAMPLES
    tf = config.EVAL.TEACHER_FORCING
    resume = config.EVAL.RESUME

    metric, full_result = benchmark.test_with_examples(
        model, times,
        num_examples=num_exp,
        teacher_forcing=tf,
        resume=resume
    )

    return metric, full_result
