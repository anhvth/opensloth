from base_dataset_preparer import BaseDatasetPreparer


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


def main():
    preparer = GemmaDatasetPreparer()
    preparer.run()


if __name__ == "__main__":
    main()
