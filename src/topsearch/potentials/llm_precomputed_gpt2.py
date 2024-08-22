""" Module that contains the class for evaluating the loss and gradients
    of a GPT2-large Sequence Classification Head Model ONLY,
    via the Huggingface Transformers and Database libraries
"""
import numpy as np
from nptyping import NDArray
from .llm import LLM
import transformers
import torch 
from datasets import load_dataset
import numpy as np
from transformers import DataCollatorWithPadding, TrainingArguments, Trainer
import tempfile

class PrecomputedGPT2(LLM):
    """
    Description
    ---------------
    Evaluate the loss and gradients of a Huggingface model, wrapped inside a Trainer, as in the example.

    Attributes
    ---------------
    trainer : transformer.trainer.Trainer
        A Huggingface wrapper for Pytorch models, datasets and arguments (such as batch size).
    """

    def __init__(self, regularizer_lambda = 1e-7, precomputed_weights_torch_file=None) -> None:
        imdb = load_dataset("imdb")
        # Create a smaller training dataset for faster training times
        small_train_dataset = imdb["train"].shuffle(seed=42).select([i for i in list(range(1024))])
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained("gpt2-large", num_labels=2)
        try: 
            import intel_extension_for_pytorch as ipex
            model.to('xpu')
            model = ipex.optimize(model)
            use_ipex=True
        except:
            use_ipex=False
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model.to(device)
            pass
        # Freeze base layers
        for param in model.base_model.parameters():
            param.requires_grad = False
        if tokenizer.pad_token is None:
            print(f'Inserting pad token')
            tokenizer.add_special_tokens({'pad_token': '<|padding|>'})
            model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.pad_token_id
        # Prepare the text inputs for the model
        def preprocess_function(examples):
            return tokenizer(examples["text"], truncation=True)
        tokenized_train = small_train_dataset.map(preprocess_function, batched=True)
        # Use data_collector to convert our samples to PyTorch tensors and concatenate them with the correct amount of padding
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        # Define a new Trainer with all the objects we constructed so far
        with tempfile.TemporaryDirectory() as tempdir:
            training_args = TrainingArguments(
                output_dir=tempdir,
                per_device_train_batch_size=16,
                gradient_accumulation_steps=64,
                per_device_eval_batch_size=16,
                use_ipex=use_ipex
            )
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_train,
                tokenizer=tokenizer,
                data_collator=data_collator,
            )
        
        self.trainer = trainer 
        self.model = model
        self.precomputed_weights = torch.load(precomputed_weights_torch_file)
        self.precomputed_weights_training_data = [_ for _ in trainer.get_train_dataloader()]
        self.n_params=2560
        self.trainable_layers_names_and_sizes=[('score.weight', torch.Size([2, 1280]))]
        self.regularizer_lambda = regularizer_lambda


    def function(self, position: NDArray) -> float:
        """ Compute the loss value at a specific configuration of weights """
        return self.function_gradient(position)[0]

    def function_gradient(self, position: NDArray) -> tuple:
        """ Compute the loss and gradient at a specific configuration of weights """
        self.set_new_weights(position) # modifies the weights of the model according to the position vector
        self.model.zero_grad()
        total_loss = 0
        for hidden_states,inputs in zip(self.precomputed_weights,self.precomputed_weights_training_data): 
            self.trainer.model.eval() # one needs this otherwise the model changes on every evaluation, due to Dropout and LayerNorm
            loss = self.compute_loss_of_precomputed_model(hidden_states,inputs)
            # append L2 regularisation
            loss += self.regularizer_lambda * self.trainer.model.score.weight.pow(2.0).sum()
            # get gradients
            self.trainer.accelerator.backward(loss)
            total_loss += loss.detach() / len(self.precomputed_weights_training_data)
        gradients = []
        # get gradients
        gradients.append(self.trainer.model.score.weight.grad.flatten().tolist())
        gradients = np.concatenate([gradients])
        return total_loss.item(), gradients.flatten()

    def compute_loss_of_precomputed_model(self,hidden_states,inputs):
        """
        Given the hidden states of one batch and input labels, compute the prediction and loss

        Sequence of commands copied from transformers.models.gpt2.modeling_gpt2.py
        """
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        logits = self.trainer.model.score(hidden_states)
        sequence_lengths = torch.eq(input_ids, self.trainer.model.config.pad_token_id).int().argmax(-1) - 1
        sequence_lengths = sequence_lengths % input_ids.shape[-1]
        sequence_lengths = sequence_lengths.to(logits.device)
        pooled_logits = logits[torch.arange(self.trainer.args.per_device_train_batch_size, device=logits.device), sequence_lengths]
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(pooled_logits.view(-1, self.trainer.model.num_labels), labels.view(-1))
        return loss

    # def compute_loss_of_precomputed_model(self,hidden_states,inputs):
    #     """
    #     Given the hidden states of one batch and input labels, compute the prediction and loss

    #     Sequence of commands copied from transformers.models.gpt2.modeling_gpt2.py
    #     """
    #     input_ids = inputs['input_ids']
    #     labels = inputs['labels']
    #     logits = self.trainer.model.score(hidden_states)
    #     sequence_lengths = torch.eq(input_ids, self.trainer.model.config.pad_token_id).int().argmax(-1) - 1
    #     sequence_lengths = sequence_lengths % input_ids.shape[-1]
    #     sequence_lengths = sequence_lengths.to(logits.device)
    #     pooled_logits = logits[torch.arange(self.trainer.args.per_device_train_batch_size, device=logits.device), sequence_lengths]
    #     loss_fct = torch.nn.CrossEntropyLoss()
    #     loss = loss_fct(pooled_logits.view(-1, self.trainer.model.num_labels), labels.view(-1))
    #     return loss
    

class PrecomputedGPT2Test(PrecomputedGPT2):
    """
    Description
    ---------------
    Evaluate the test loss and gradients of a Huggingface model, wrapped inside a Trainer, as in the example.

    Attributes
    ---------------
    trainer : transformer.trainer.Trainer
        A Huggingface wrapper for Pytorch models, datasets and arguments (such as batch size).
    """

    def __init__(self, regularizer_lambda = 1e-7, precomputed_weights_torch_file=None) -> None:
        imdb = load_dataset("imdb")
        # Create a smaller training dataset for faster training times
        small_test_dataset = imdb["test"].shuffle(seed=42).select([i for i in list(range(300))])
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained("gpt2-large", num_labels=2)
        try: 
            import intel_extension_for_pytorch as ipex
            model.to('xpu')
            model = ipex.optimize(model)
            use_ipex=True
        except:
            use_ipex=False
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model.to(device)
            pass
        # Freeze base layers
        for param in model.base_model.parameters():
            param.requires_grad = False
        if tokenizer.pad_token is None:
            print(f'Inserting pad token')
            tokenizer.add_special_tokens({'pad_token': '<|padding|>'})
            model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.pad_token_id
        # Prepare the text inputs for the model
        def preprocess_function(examples):
            return tokenizer(examples["text"], truncation=True)
        tokenized_test = small_test_dataset.map(preprocess_function, batched=True)
        # Use data_collector to convert our samples to PyTorch tensors and concatenate them with the correct amount of padding
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        # Define a new Trainer with all the objects we constructed so far
        with tempfile.TemporaryDirectory() as tempdir:
            training_args = TrainingArguments(
                output_dir=tempdir,
                per_device_train_batch_size=15,
                gradient_accumulation_steps=20,
                use_ipex=use_ipex
            )
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_test,
                tokenizer=tokenizer,
                data_collator=data_collator,
            )
        
        self.trainer = trainer 
        self.model = model
        self.precomputed_weights = torch.load(precomputed_weights_torch_file)
        self.precomputed_weights_training_data = [_ for _ in trainer.get_train_dataloader()]
        self.n_params=2560
        self.trainable_layers_names_and_sizes=[('score.weight', torch.Size([2, 1280]))]
        self.regularizer_lambda = regularizer_lambda
