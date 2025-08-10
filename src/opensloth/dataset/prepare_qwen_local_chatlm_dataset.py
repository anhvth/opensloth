
# NOTE: Avoid global unsloth imports to prevent GPU registry issues  
# Unsloth will be imported on-demand when needed during dataset preparation
from opensloth._debug_dataloader import debug_chat_dataloader_for_training_markdown

from .base_dataset_preparer import BaseDatasetPreparer


class QwenLocalDatasetPreparer(BaseDatasetPreparer):
    """Dataset preparer for Qwen models using local chat datasets."""
    
    def get_description(self) -> str:
        return "Prepare Qwen dataset with tokenization and formatting from local files."
    
    def get_default_tokenizer(self) -> str:
        return 'unsloth/Qwen3-32B'
    
    def get_default_chat_template(self) -> str:
        return "qwen-3"
    
    def get_default_dataset_name(self) -> str:
        return '../../TRANSLATE_UI/processed_messages.json'
    
    def get_default_instruction_part(self) -> str:
        return '<start_of_turn>user\n'
    
    def get_default_response_part(self) -> str:
        return '<start_of_turn>assistant\n'
    
    def add_custom_arguments(self, parser):
        """Add local dataset specific arguments."""
        parser.add_argument('--input_file', '-i', type=str, 
                          default='../../TRANSLATE_UI/processed_messages.json',
                          help='Input JSON file with messages')
    
    def add_custom_config_entries(self):
        """Add custom config entries."""
        if hasattr(self.args, 'input_file'):
            self.config_dict["Input file"] = self.args.input_file
    
    def load_dataset(self):
        """Load dataset from local file."""
        # Override dataset_name with input_file for local loading
        if hasattr(self.args, 'input_file'):
            self.args.dataset_name = self.args.input_file
        
        return super().load_dataset()
    
    def post_process_text(self, text: str) -> str:
        """Post-process text for Qwen models."""
        # Remove thinking tokens for instruct models
        if 'instruct-2507' in self.args.tok_name.lower():
            text = text.replace('<think>\n\n</think>\n\n', '')
        return text
    
    def debug_visualization(self, data):
        """Use markdown debug visualization for local datasets."""
        if self.args.debug > 0:
            print(f"[INFO] Debug mode enabled. Dumping {self.args.debug} samples to HTML and terminal...")
            data.set_format(type="torch", columns=["input_ids", "labels"])
            from torch.utils.data import DataLoader
            dataloader = DataLoader(data, batch_size=1, shuffle=False)
            debug_chat_dataloader_for_training_markdown(dataloader, self.tokenizer, n_example=self.args.debug)


def main():
    preparer = QwenLocalDatasetPreparer()
    preparer.run()


if __name__ == "__main__":
    main()
