class WeightPathNotValidException(Exception):
    def __init__(self, model_name: str):
        self.message = f"The weight path of {model_name} is not a directory. Check the weight path in the init method."
        super().__init__(self.message)
