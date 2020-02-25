import os

import test_utils


EXAMPLES_ROOT = test_utils.EXAMPLES_ROOT


def test_1():
    root_dir = os.path.join(EXAMPLES_ROOT, 'mnist')

    output_evaluator = test_utils.TemplateOutputEvaluator(
        b'''\
Device: @numpy
# unit: 10
# Minibatch-size: 100
# epoch: 1

epoch       main/loss   validation/main/loss  main/accuracy  validation/main/accuracy  elapsed_time
0                       {b0                 }                {d0                     } {e0        }
1           {a1       } {b1                 } {c1          } {d1                     } {e1        }
''',  # NOQA
        b0=(float, lambda x: 0.0 < x),
        d0=(float, lambda x: 0.00 <= x <= 1.00),
        e0=(float, lambda x: 0. < x < 100.),
        a1=(float, lambda x: 0.6 < x < 1.5),
        b1=(float, lambda x: 0.3 < x < 0.6),
        c1=(float, lambda x: 0.62 < x < 0.82),
        d1=(float, lambda x: 0.83 < x < 0.98),
        e1=(float, lambda x: 0. < x < 100.),
    )

    with test_utils.ExampleRunner(root_dir) as r:

        r.run(
            'train_mnist.py',
            [
                '--epoch', '1',
                '--unit', '10',
            ],
            output_evaluator=output_evaluator)
