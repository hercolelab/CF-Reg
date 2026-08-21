import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import TensorDataset

import main as entrypoint


class RecordingTrainer:
    def __init__(self):
        self.fit_calls = []
        self.test_calls = []

    def fit(self, *args, **kwargs):
        self.fit_calls.append((args, kwargs))

    def test(self, *args, **kwargs):
        self.test_calls.append((args, kwargs))


class MainDataFlowTests(unittest.TestCase):
    def setUp(self):
        self.train_set = TensorDataset(
            torch.tensor([[1.0], [2.0]]),
            torch.tensor([0, 1]),
        )
        self.validation_set = TensorDataset(
            torch.tensor([[3.0], [4.0]]),
            torch.tensor([1, 0]),
        )
        self.test_set = TensorDataset(
            torch.tensor([[99.0]]),
            torch.tensor([1]),
        )
        self.loader_config = {"batch_size": 2, "shuffle": True}
        self.classifier = object()

    def _record_loaders(self):
        created_loaders = []

        def create_loader(dataset, **config):
            loader = SimpleNamespace(dataset=dataset, config=config)
            created_loaders.append(loader)
            return loader

        return created_loaders, create_loader

    def test_tuning_fits_train_and_validation_without_using_test(self):
        fit_set, holdout_test_set = entrypoint.select_datasets_for_mode(
            train_set=self.train_set,
            validation_set=self.validation_set,
            test_set=self.test_set,
            tuning=True,
            early_stopping_enabled=True,
        )
        self.assertIs(fit_set, self.train_set)
        self.assertIsNone(holdout_test_set)

        trainer = RecordingTrainer()
        created_loaders, create_loader = self._record_loaders()

        with patch.object(entrypoint, "DataLoader", side_effect=create_loader):
            entrypoint.fit_and_evaluate(
                trainer=trainer,
                classifier=self.classifier,
                fit_set=fit_set,
                validation_set=self.validation_set,
                test_set=holdout_test_set,
                loader_config=self.loader_config,
                tuning=True,
                early_stopping_enabled=True,
            )

        self.assertEqual(len(created_loaders), 2)
        self.assertIs(created_loaders[0].dataset, self.train_set)
        self.assertIs(created_loaders[1].dataset, self.validation_set)
        self.assertNotIn(self.test_set, [loader.dataset for loader in created_loaders])
        self.assertTrue(created_loaders[0].config["shuffle"])
        self.assertFalse(created_loaders[1].config["shuffle"])

        self.assertEqual(len(trainer.fit_calls), 1)
        fit_args, fit_kwargs = trainer.fit_calls[0]
        self.assertEqual(fit_kwargs, {})
        self.assertEqual(fit_args, (self.classifier, created_loaders[0], created_loaders[1]))
        self.assertEqual(trainer.test_calls, [])

    def test_final_run_fits_train_validation_union_then_tests_holdout(self):
        final_train_set, holdout_test_set = entrypoint.select_datasets_for_mode(
            train_set=self.train_set,
            validation_set=self.validation_set,
            test_set=self.test_set,
            tuning=False,
            early_stopping_enabled=False,
        )
        self.assertIs(holdout_test_set, self.test_set)

        trainer = RecordingTrainer()
        created_loaders, create_loader = self._record_loaders()

        with patch.object(entrypoint, "DataLoader", side_effect=create_loader):
            entrypoint.fit_and_evaluate(
                trainer=trainer,
                classifier=self.classifier,
                fit_set=final_train_set,
                validation_set=self.validation_set,
                test_set=holdout_test_set,
                loader_config=self.loader_config,
                tuning=False,
                early_stopping_enabled=False,
            )

        expected_features = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        expected_targets = torch.tensor([0, 1, 1, 0])
        self.assertTrue(torch.equal(final_train_set.tensors[0], expected_features))
        self.assertTrue(torch.equal(final_train_set.tensors[1], expected_targets))
        self.assertFalse(torch.any(final_train_set.tensors[0] == 99.0))

        self.assertEqual(len(created_loaders), 2)
        self.assertIs(created_loaders[0].dataset, final_train_set)
        self.assertIs(created_loaders[1].dataset, self.test_set)
        self.assertNotIn(self.validation_set, [loader.dataset for loader in created_loaders])
        self.assertTrue(created_loaders[0].config["shuffle"])
        self.assertFalse(created_loaders[1].config["shuffle"])

        self.assertEqual(len(trainer.fit_calls), 1)
        fit_args, fit_kwargs = trainer.fit_calls[0]
        self.assertEqual(fit_kwargs, {})
        self.assertEqual(fit_args, (self.classifier, created_loaders[0]))

        self.assertEqual(len(trainer.test_calls), 1)
        test_args, test_kwargs = trainer.test_calls[0]
        self.assertEqual(test_args, (self.classifier,))
        self.assertIs(test_kwargs["dataloaders"], created_loaders[1])
        self.assertIsNone(test_kwargs["ckpt_path"])

    def test_final_early_stopping_validates_then_tests_best_checkpoint(self):
        fit_set, holdout_test_set = entrypoint.select_datasets_for_mode(
            train_set=self.train_set,
            validation_set=self.validation_set,
            test_set=self.test_set,
            tuning=False,
            early_stopping_enabled=True,
        )
        self.assertIs(fit_set, self.train_set)
        self.assertIs(holdout_test_set, self.test_set)

        trainer = RecordingTrainer()
        created_loaders, create_loader = self._record_loaders()

        with patch.object(entrypoint, "DataLoader", side_effect=create_loader):
            entrypoint.fit_and_evaluate(
                trainer=trainer,
                classifier=self.classifier,
                fit_set=fit_set,
                validation_set=self.validation_set,
                test_set=holdout_test_set,
                loader_config=self.loader_config,
                tuning=False,
                early_stopping_enabled=True,
            )

        self.assertEqual(len(created_loaders), 3)
        self.assertIs(created_loaders[0].dataset, self.train_set)
        self.assertIs(created_loaders[1].dataset, self.validation_set)
        self.assertIs(created_loaders[2].dataset, self.test_set)
        self.assertTrue(created_loaders[0].config["shuffle"])
        self.assertFalse(created_loaders[1].config["shuffle"])
        self.assertFalse(created_loaders[2].config["shuffle"])
        self.assertTrue(self.loader_config["shuffle"])

        self.assertEqual(len(trainer.fit_calls), 1)
        fit_args, fit_kwargs = trainer.fit_calls[0]
        self.assertEqual(fit_kwargs, {})
        self.assertEqual(
            fit_args,
            (self.classifier, created_loaders[0], created_loaders[1]),
        )

        self.assertEqual(len(trainer.test_calls), 1)
        test_args, test_kwargs = trainer.test_calls[0]
        self.assertEqual(test_args, (self.classifier,))
        self.assertIs(test_kwargs["dataloaders"], created_loaders[2])
        self.assertEqual(test_kwargs["ckpt_path"], "best")

    def test_final_run_requires_a_holdout_test_set(self):
        final_train_set = entrypoint.merge_train_validation_sets(
            self.train_set,
            self.validation_set,
        )

        with self.assertRaisesRegex(ValueError, "holdout test set"):
            entrypoint.fit_and_evaluate(
                trainer=RecordingTrainer(),
                classifier=self.classifier,
                fit_set=final_train_set,
                validation_set=self.validation_set,
                test_set=None,
                loader_config=self.loader_config,
                tuning=False,
                early_stopping_enabled=False,
            )

    def test_merge_requires_matching_tensor_structures(self):
        validation_with_one_tensor = TensorDataset(torch.tensor([[3.0]]))

        with self.assertRaisesRegex(ValueError, "same number of tensors"):
            entrypoint.merge_train_validation_sets(
                self.train_set,
                validation_with_one_tensor,
            )

    def test_early_stopping_creates_a_matching_best_checkpoint(self):
        callbacks = entrypoint.get_callbacks(
            early_stop_enable=True,
            monitor="validation/accuracy",
            patience=7,
            verbose=False,
            mode="max",
        )

        self.assertEqual(len(callbacks), 2)
        early_stopping = next(
            callback for callback in callbacks if isinstance(callback, EarlyStopping)
        )
        checkpoint = next(
            callback for callback in callbacks if isinstance(callback, ModelCheckpoint)
        )
        self.assertEqual(early_stopping.monitor, "validation/accuracy")
        self.assertEqual(early_stopping.mode, "max")
        self.assertEqual(early_stopping.patience, 7)
        self.assertEqual(checkpoint.monitor, early_stopping.monitor)
        self.assertEqual(checkpoint.mode, early_stopping.mode)
        self.assertEqual(checkpoint.save_top_k, 1)

    def test_patience_does_not_enable_early_stopping_by_itself(self):
        callbacks = entrypoint.get_callbacks(
            early_stop_enable=False,
            patience=7,
        )

        self.assertIsNone(callbacks)


if __name__ == "__main__":
    unittest.main()
