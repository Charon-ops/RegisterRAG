class LoadNotInitializedException(Exception):
    """
    Exception raised when the load method of the embedding class is not initialized correctly
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.message = f"The load method of model '{model_name}' is not initialized correctly. Check the init method in the embedding class"
        super().__init__(self.message)

    def __str__(self):
        return self.message
