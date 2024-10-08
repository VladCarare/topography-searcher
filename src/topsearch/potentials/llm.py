""" Module that contains the class for evaluating the loss and gradients
    of a Large Language Model (LLM) via the Huggingface Transformers and Database libraries"""


""" Example of use:
import torch 

try: 
    import intel_extension_for_pytorch as ipex
    print('Found IPEX package for running models on Intel GPUs.')
except:
    pass


# Load data
from datasets import load_dataset
imdb = load_dataset("imdb")

# Create a smaller training dataset for faster training times
small_train_dataset = imdb["train"].shuffle(seed=42).select([i for i in list(range(16))])
small_test_dataset = imdb["test"].shuffle(seed=42).select([i for i in list(range(10))])

# Set tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2-large")


# Define our base model:
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("gpt2-large", num_labels=2)
model.to('xpu')
model = ipex.optimize(model)
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
tokenized_test = small_test_dataset.map(preprocess_function, batched=True)


# Use data_collector to convert our samples to PyTorch tensors and concatenate them with the correct amount of padding
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# Define a new Trainer with all the objects we constructed so far
from transformers import TrainingArguments, Trainer

repo_name = "test"
training_args = TrainingArguments(
    output_dir=repo_name,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=0,
    use_ipex=True,
)
trainer_just_head = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

from topsearch.potentials.llm import LLM
import numpy as np

llm = LLM(trainer_just_head,regularizer_lambda=1e-7)
llm.function_gradient(np.zeros(2560))
"""
import numpy as np
from nptyping import NDArray
from .potential import Potential
import transformers
import torch 

class LLM(Potential):
    """
    Description
    ---------------
    Evaluate the loss and gradients of a Huggingface model, wrapped inside a Trainer, as in the example.

    Attributes
    ---------------
    trainer : transformer.trainer.Trainer
        A Huggingface wrapper for Pytorch models, datasets and arguments (such as batch size).
    """

    def __init__(self, trainer: transformers.trainer.Trainer, regularizer_lambda = 0.01) -> None:
        self.n_params = trainer.get_num_trainable_parameters()
        if self.n_params > 1e4:
            raise ValueError(f"Number of trainable parameters is {self.n_params}. We can only accept values smaller than 10,000.\n Make sure you use\n\"for param in model.base_model.parameters():\n param.requires_grad = False\"\n or select a model with a smaller head.")
        self.trainer = trainer
        self.model = trainer.model
        self.trainable_layers_names_and_sizes = [(name,param.size()) for (name,param) in self.model.named_parameters() if param.requires_grad]
        # get data
        self.inputs = trainer.get_train_dataloader()
        self.num_examples = trainer.num_examples(self.inputs)
        if self.num_examples!=trainer.args.gradient_accumulation_steps*trainer.args.per_device_train_batch_size:
            raise ValueError(f"At the moment we cannot evaluate mini-batches. Make sure the size of the training batch matches the size of data\n or that gradient_accumulation_steps*per_device_train_batch_size = train_size")
        self.regularizer_lambda = regularizer_lambda

    def function(self, position: NDArray) -> float:
        """ Compute the loss value at a specific configuration of weights """
        self.set_new_weights(position)
        loss = 0
        for inputs in self.inputs: # run over gradient accumulation steps, if any
            loss += self.trainer.prediction_step(self.model,inputs,prediction_loss_only=True)[0]
            # append L2 regularisation
            for layer_name, _ in self.trainable_layers_names_and_sizes:
                layer = self.find_layer(self.model,layer_name) # get layer by name 
                loss += self.regularizer_lambda * layer.pow(2.0).sum()
        return loss.item() / self.trainer.args.gradient_accumulation_steps 

    def function_gradient(self, position: NDArray) -> tuple:
        """ Compute the loss and gradient at a specific configuration of weights """
        self.set_new_weights(position) # modifies the weights of the model according to the position vector
        self.model.zero_grad()
        total_loss = 0
        for inputs in self.inputs: # run over gradient accumulation steps, if any
            self.model.eval() # one needs this otherwise the model changes on every evaluation, due to Dropout and LayerNorm
            loss = self.trainer.compute_loss(self.model, inputs)
            # append L2 regularisation
            for layer_name, _ in self.trainable_layers_names_and_sizes:
                layer = self.find_layer(self.model,layer_name) # get layer by name 
                loss += self.regularizer_lambda * layer.pow(2.0).sum()
            # get gradients
            self.trainer.accelerator.backward(loss)
            total_loss += loss.detach() / self.trainer.args.gradient_accumulation_steps
        gradients = []
        # get gradients
        for layer_name, _ in self.trainable_layers_names_and_sizes:
            layer = self.find_layer(self.model,layer_name) # get layer by name 
            gradients.append(layer.grad.flatten().tolist())
        gradients = np.concatenate([gradients])
        return total_loss.item(), gradients.flatten()

    def gradient(self, position: NDArray) -> NDArray:
        """ Compute the gradient at a specific configuration of weights """
        _, gradients = self.function_gradient(position)
        return gradients

    def hessian(self, position: NDArray) -> NDArray:
        raise(ValueError('Hessian not yet implemented'))

    def set_new_weights(self, position: NDArray) -> None:
        """ Compute the loss value at a specific configuration of weights """
        if len(position) != self.n_params:
            raise ValueError(f"Number of elements of the position array {len(position)} needs to be equal to the number of trainable parameters of the network {self.n_params}.")
        # resize position parameters to the sizes of the trainable layers, and reset the corresponding parameters
        position = (_ for _ in position) 
        for layer_name, size in self.trainable_layers_names_and_sizes:
            layer = self.find_layer(self.model,layer_name) # get layer by name 
            num_weights = layer.numel()
            new_weights = torch.Tensor([next(position) for _ in range(num_weights)]).reshape(size).to(self.model.device)
            layer.data = new_weights

    @staticmethod
    def find_layer(model: torch.nn.Module, name: str) -> torch.nn.Module:
        """ Find a model layer by name
        """
        sub_modules = name.split('.')
        current_module = model
        try:
            for sub_module in sub_modules:
                current_module = getattr(current_module, sub_module)
            return current_module
        except AttributeError:
            return None
        
