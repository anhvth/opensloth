from .base_dataset_preparer import BaseDatasetPreparer


class GemmaDatasetPreparer(BaseDatasetPreparer):
    """Dataset preparer for Gemma models."""
    
    def get_description(self) -> str:
        return "Prepare Gemma dataset with tokenization and formatting."
    
    def get_default_tokenizer(self) -> str:
        return 'unsloth/gemma-3-4b-it'
    
    def get_default_chat_template(self) -> str:
        return "gemma-3"
    
    def get_default_dataset_name(self) -> str:
        return 'mlabonne/FineTome-100k'
    
    def get_default_instruction_part(self) -> str:
        return "<start_of_turn>user\n"
    
    def get_default_response_part(self) -> str:
        return '<start_of_turn>model\n'
    
    def add_custom_arguments(self, parser):
        """Add Gemma-specific arguments."""
        parser.add_argument('--input_file', '-i', type=str, default=None,
                          help='Input JSON file with messages (alternative to dataset_name)')
    
    def add_custom_config_entries(self):
        """Add Gemma-specific config entries."""
        if hasattr(self.args, 'input_file') and self.args.input_file:
            self.config_dict["Input file"] = self.args.input_file
    
    def load_dataset(self):
        """Load dataset from HuggingFace or local file."""
        # If input_file is specified, use it instead of dataset_name
        if hasattr(self.args, 'input_file') and self.args.input_file:
            self.args.dataset_name = self.args.input_file
        
        return super().load_dataset()


def main():
    preparer = GemmaDatasetPreparer()
    preparer.run()


if __name__ == "__main__":
    main()
