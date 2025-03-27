class DotDict:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, DotDict(value))
            else:
                setattr(self, key, value)

    def __getattr__(self, name):
        return self.__dict__.get(name)

    def __setattr__(self, name, value):
        if isinstance(value, dict):
            self.__dict__[name] = DotDict(value)
        else:
            self.__dict__[name] = value

    def __delattr__(self, name):
        del self.__dict__[name]

    def to_dict(self):
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, DotDict):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
