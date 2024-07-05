# import pytest
from transformers import (
        GPT2Config,
        GPT2LMHeadModel,
        TrainingArguments, 
        Trainer
    )
import tempfile
import torch 
from topsearch.potentials import llm
import numpy as np
torch.use_deterministic_algorithms(True)
torch.manual_seed(0)
import random
random.seed(0)

class RepeatDataset:
    """Class copied from transformers/trainer_tests.py"""
    def __init__(self, x, length=64):
        self.x = x
        self.length = length
    def __len__(self):
        return self.length
    def __getitem__(self, i):
        return {"input_ids": self.x, "labels": self.x}

def test_llm_weight_setting():
    """
    Set model's head weights to different values and check for reproducibility and expected loss and gradients.
    """
    # create a very small model akin to what one may encounter in production
    config = GPT2Config(vocab_size=10, n_positions=20, n_embd=2, n_layer=1, n_head=1)
    tiny_gpt2 = GPT2LMHeadModel(config)
    # create random (but determistic) training set
    x = torch.randint(0, 10, (20,))
    train_dataset = RepeatDataset(x,length=10)
    # the Trainer object needs a working directory, so create a temporary one
    with tempfile.TemporaryDirectory() as tmpdir:
        # Trainer without inf/nan filter
        args = TrainingArguments(
            tmpdir, learning_rate=1e-13, per_device_train_batch_size=10, gradient_accumulation_steps=1, 
            logging_steps=5, logging_nan_inf_filter=False, use_cpu=True, num_train_epochs=1
        )
        trainer = Trainer(tiny_gpt2, args, train_dataset=train_dataset)
        # disable gradients for all but some weights - as one would do when finetuning
        for param in trainer.model.parameters():
            param.requires_grad = False
        trainer.model.lm_head.weight.requires_grad = True
        # set the model in .eval mode - this ensures some components, such as LayerNorm, do not change
        trainer.model.eval()
        # initialise the TopSearch interface
        test_potential=llm.LLM(trainer)
        # test the 3 functions 3 times, for 2 random seeds.
        trainer.model.eval()
        np.random.seed(123)
        new_layer_data =  np.random.rand(20)
        loss = test_potential.function(new_layer_data)
        loss_gradients = test_potential.function_gradient(new_layer_data)
        gradients = test_potential.gradient(new_layer_data)
        np.random.seed(124)
        new_layer_data =  np.random.rand(20)
        loss2 = test_potential.function(new_layer_data)
        loss_gradients2 = test_potential.function_gradient(new_layer_data)
        gradients2 = test_potential.gradient(new_layer_data)
        np.random.seed(123)
        new_layer_data =  np.random.rand(20)
        loss3 = test_potential.function(new_layer_data)
        loss_gradients3 = test_potential.function_gradient(new_layer_data)
        gradients3 = test_potential.gradient(new_layer_data)
        assert loss==loss_gradients[0]
        assert loss2==loss_gradients2[0]
        assert loss3==loss_gradients3[0]
        assert loss==loss3
        assert np.all(gradients==loss_gradients[1])
        assert np.all(gradients2==loss_gradients2[1])
        assert np.all(gradients3==loss_gradients3[1])
        assert np.all(gradients==gradients3)
        assert not np.all(gradients==gradients2)
        expected_gradients = np.array([-0.03265307,  0.03265312, -0.05189211,  0.05189213,  0.06167838,
        -0.06167835,  0.11434107, -0.11434112, -0.0625431 ,  0.06254306,
         0.09921123, -0.09921129,  0.01692764, -0.01692764, -0.15870528,
         0.15870537,  0.02924541, -0.02924554, -0.01912324,  0.01912324])
        expected_gradients2 = np.array([-0.05143911,  0.05143914, -0.03922105,  0.03922106,  0.0161558 ,
        -0.01615578, -0.01927603,  0.01927605, -0.03995134,  0.03995132,
        -0.04588147,  0.04588142,  0.00554584, -0.00554584,  0.19774836,
        -0.19774836, -0.05518682,  0.05518683,  0.0302661 , -0.03026609])
        expected_loss = 2.333463430404663
        expected_loss2 = 2.332834243774414
        assert loss == expected_loss
        assert loss2 == expected_loss2
        # The expected_gradients are defined up the the 8th decimal place so use np.allclose
        assert np.allclose(gradients,expected_gradients,atol=1e-08)
        assert np.allclose(gradients2,expected_gradients2,atol=1e-08)
        # Ideally we would test against the .train() method of Trainer. But the .train() method invokes training_step(),
        # which puts the model into .training mode, which gives non-deterministic results because of dropout and layernorm.


def test_llm_gradient_accumulation():
    """
    Changing per_device_train_batch_size to 5 and gradient_accumulation_steps to 2 and 
    comparing to the results of the model with per_device_train_batch_size 10 and gradient_accumulation_steps 1
    """
    # create a very small model akin to what one may encounter in production
    config = GPT2Config(vocab_size=10, n_positions=20, n_embd=2, n_layer=1, n_head=1)
    tiny_gpt2 = GPT2LMHeadModel(config)
    # create random (but determistic) training set
    x = torch.randint(0, 10, (20,))
    train_dataset = RepeatDataset(x,length=10)
    # the Trainer object needs a working directory, so create a temporary one
    with tempfile.TemporaryDirectory() as tmpdir:
        # Trainer without inf/nan filter
        args = TrainingArguments(
            tmpdir, learning_rate=1e-13, per_device_train_batch_size=5, gradient_accumulation_steps=2, 
            logging_steps=5, logging_nan_inf_filter=False, use_cpu=True, num_train_epochs=1
        )
        trainer = Trainer(tiny_gpt2, args, train_dataset=train_dataset)
        # disable gradients for all but some weights - as one would do when finetuning
        for param in trainer.model.parameters():
            param.requires_grad = False
        trainer.model.lm_head.weight.requires_grad = True
        # set the model in .eval mode - this ensures some components, such as LayerNorm, do not change
        trainer.model.eval()
        # initialise the TopSearch interface
        test_potential=llm.LLM(trainer)
        # test the 3 functions 3 times, for 2 random seeds.
        trainer.model.eval()
        np.random.seed(123)
        new_layer_data =  np.random.rand(20)
        loss, gradients = test_potential.function_gradient(new_layer_data)
        expected_gradients = np.array([-0.03265307,  0.03265312, -0.05189211,  0.05189213,  0.06167838,
        -0.06167835,  0.11434107, -0.11434112, -0.0625431 ,  0.06254306,
         0.09921123, -0.09921129,  0.01692764, -0.01692764, -0.15870528,
         0.15870537,  0.02924541, -0.02924554, -0.01912324,  0.01912324])
        expected_loss = 2.333463430404663
        # I'm observing that using gradient accumulation steps results in digits changed in loss and gradients in the 7th decimal place
        # likely because of the extra division operations needing to be performed maybe. I am seeing the same thing with the 
        # .train() methods, so it is a feature of the transformers package as well.
        assert np.allclose(loss,expected_loss,atol=1e-7)
        assert np.allclose(gradients,expected_gradients,atol=1e-07)