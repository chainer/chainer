from chainer import testing
from chainer.training import extension as extension_module


def check_iteration_aware(
        trigger,
        before_training_callback=None,
        n_extensions=3,
        max_iterations=10,
        iter_per_epoch=5,
        extension_priority=extension_module.PRIORITY_READER):

    # Register n extensions with a single trigger instance
    # and check to see if the trigger is NOT called for each extension.

    iter_per_epoch = 5

    # Create extensions
    extension_epoch_details = [[] for _ in range(n_extensions)]
    extensions = []

    def create_extension(i):
        def extension(t):
            extension_epoch_details[i].append(t.updater.epoch_detail)
        return extension

    extensions = [create_extension(i) for i in range(n_extensions)]

    # Prepare the trainer
    trainer = testing.get_trainer_with_mock_updater(
        stop_trigger=(max_iterations, 'iteration'),
        iter_per_epoch=iter_per_epoch)

    for i, extension in enumerate(extensions):
        trainer.extend(
            extension,
            name='ext{}'.format(i),
            trigger=trigger,
            priority=extension_priority)

    if before_training_callback is not None:
        before_training_callback(trainer)

    # Run the trainer
    trainer.run()

    # Each extension must have been triggered in the same timings
    for i in range(1, n_extensions):
        assert extension_epoch_details[i] == extension_epoch_details[0]
