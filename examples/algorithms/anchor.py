import torch
from models.initializer import initialize_model
from algorithms.single_model_algorithm import SingleModelAlgorithm
from wilds.common.utils import split_into_groups
from utils import concat_input

class Anchor(SingleModelAlgorithm):
    """
    Deep Anchor Regression
    """
    def __init__(self, config, d_out, grouper, loss, metric, n_train_steps):
        # check config
        assert config.train_loader == 'group'
        # assert config.uniform_over_groups
        # assert config.distinct_groups
        # initialize models
        model = initialize_model(config, d_out=d_out).to(config.device)

        # initialize module
        super().__init__(
            config=config,
            model=model,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )
        # algorithm hyperparameters
        self.anchor_penalty = config.anchor_penalty
        # set model components
        self.logged_fields.append('penalty')

    def objective(self, results):
        _, group_indices, _ = split_into_groups(results['g'])
        avg_loss = 0.
        penalty = 0.

        for i_group in group_indices: # Each element of group_indices is a list of indices
            residuals = results['y_pred'][i_group] - results['y_true'][i_group]
            if len(i_group)>0:
                if self.is_training: # Penalties only make sense when training
                    penalty += len(i_group) * residuals.mean()**2
                avg_loss += len(i_group) * (residuals**2).mean()

        self.save_metric_for_logging(results, 'penalty', penalty)
        return avg_loss +  (self.anchor_penalty - 1) * penalty
