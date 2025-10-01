import numpy as np
import json

class Dependent:

    def __init__(self, *value, parameters = {}, label = '', dtype = np.float64):
        
        self.parameters = parameters
        
        if len(value) == 0:
            dimensions = [len(param_list) for param_name, param_list in self.parameters.items()]
            self.value = np.full(dimensions, np.nan, dtype = dtype)
        elif len(value) == 1:
            self.value = np.array(value[0], dtype = dtype)
        else:
            raise Exception('Only 0 or 1 positional argument allowed.')
            
        self.label = label

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        self._value = np.array(val)
        
    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, parameters_dict_of_lists):
        self._parameters = {}
        for key, value in parameters_dict_of_lists.items():
            
            # Force scalars to be in a 1-dimension ndarray (not 0-dimension)
            if np.isscalar(value):
                self._parameters[key] = np.array([value])
                
            # 1-dimension data are of, just force to be in a 1-dimension ndarray
            elif isinstance(value, (list, np.ndarray)):
                self._parameters[key] = np.array(value)  #impose np.array as type for params
                
            else:
                raise Exception('parameter must be a scalar or a vector')
                
    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value):
        self._label = value
         
    @property
    def dtype(self):
        return self._value.dtype
    
    @property
    def shape(self):
        return self._value.shape
    
    def __repr__(self):
        value = self.value
        parameters = self.parameters
        string = f'{self.label} {self.shape} {self.dtype} Dependent\n{value}\n'
        
        # Pretty print the parameters
        max_key_len = max(len(key) for key in parameters.keys()) if parameters else 0 # Find the maximum key length for alignment
        string += '-' * (max_key_len+1) + '\n'
        for key, value in parameters.items():
            # Use an f-string with padding
            # :<{max_key_len} means left-align in a field of width max_key_len
            string += f"{key:<{max_key_len}}: {value}\n"
        return string
        
    def __getitem__(self, key):
        value = self.value[self._to_slices(key)]
        parameters = {param_name: self.parameters[param_name][key[param_index]]for param_index, param_name in enumerate(self.parameters.keys())}
        return Dependent(value, parameters = parameters, label = self.label, dtype = self.dtype)
    
    def __setitem__(self, key, value):
       self.value[self._to_slices(key)] = value
        # self.parameters = {param_name: self.parameters[param_name][key[param_index]]for param_index, param_name in enumerate(self.parameters.keys())}
        
        
    def _to_slices(self, key):
        """Converts all indices in a multi-dimensional key to slices using a generator expression."""
        if isinstance(key, int):
            return slice(key, key + 1)
        
        if isinstance(key, slice):
            return key
        
        # Use a generator to convert each element and create a new tuple
        return tuple(slice(item, item + 1) if isinstance(item, int) else item for item in key)

    def clone(self):
        return Dependent(self.value, parameters = self.parameters, label = self.label, dtype = self.dtype)

    def squeeze(self):
        shape = self.value.shape
        param_names = list(self.parameters.keys())
        parameters = {}
        key = []
        for dim_index, dim_len in enumerate(shape):
            if dim_len > 1:
                param_name = param_names[dim_index]
                parameters[param_name] = self.parameters[param_name]
                key.append(slice(dim_len))
            else:
                key.append(0)
                
        value = self.value[tuple(key)]
        return Dependent(value, parameters = parameters, label = self.label, dtype = self.dtype)
            
    @classmethod
    def from_dict(cls, data_dict):
        parameters = data_dict['parameters']
        label = data_dict['label']
        if 'value' in data_dict.keys():
            value = np.array(data_dict['value'], dtype = np.float64)
            dtype = np.float64
        else:
            value = (np.array(data_dict['value_re'], dtype=np.float64) 
                     + 1j * np.array(data_dict['value_im'], dtype=np.float64)).astype(np.complex128)
            dtype = np.complex128

        return cls(value, label = label, parameters = parameters, dtype = dtype)

    def to_dict(self) -> dict:
        """
        Converts the entire Dependent into a formatted dict.
        """
        # 1. Convert the Dependent into a standard dictionary
        data_dict = {
            "label": self.label,
            "parameters": {param: param_array.tolist()for param, param_array in self.parameters.items()}
            }
    
        value = self.value
        if (np.isreal(value)).all():
            data_dict['value'] = value.tolist()
        else:
            data_dict['value_re'] = np.real(value).tolist()
            data_dict['value_im'] = np.imag(value).tolist()
        
        return data_dict

    @classmethod
    def from_json(cls, json_string):
        data_dict = json.loads(json_string)
        return cls.from_dict(data_dict)

    def to_json(self) -> str:
        """
        Converts the entire Dependent into a formatted JSON string.
        """
        data_dict = self.to_dict()
        return json.dumps(data_dict, indent=2)
       
    @classmethod
    def save_dict_of_dependents(cls, dict_of_dependents, json_file_path):
        
        def custom_encoder(cls, obj):
            
            # print(isinstance(obj, DependentClass))
            # for Dependents
            if isinstance(obj, cls):
                return obj.to_dict() # Convert NumPy array to list
            # for dict of Dependents
            if isinstance(obj, dict) and all(isinstance(value, cls) for value in obj.values()):
                return {key: value.to_dict() for key, value in obj.items()}
            raise TypeError("unhandled type") # Raise error for unhandled types
        
        with open(json_file_path, 'w') as json_file:
            json.dump(dict_of_dependents, json_file, default=lambda obj:custom_encoder(cls, obj), indent=2)
            
    @classmethod
    def load_dict_of_dependents(cls, json_file_path):
        try:
            with open(json_file_path, 'r') as json_file:
                loaded_data = json.load(json_file)
                dict_of_dependents = {key: cls.from_dict(value) for key, value in loaded_data.items()}
                return dict_of_dependents
                
        except FileNotFoundError:
            print(f"Error: The file {json_file_path} was not found.")
        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON from {json_file_path}. The file content is invalid.")
        




if __name__ == "__main__":
    
    print('*** Create an empty Dependent) ***')
    d1 = Dependent()
    print(d1)
    
    print('*** Create a nan Dependent (parameters only) ***')
    d2 = Dependent(parameters = {'p1': [1, 2], 'p2': [10, 20, 30]})
    print(d2)
    
    print('*** Create a float64 Dependent (default type) ***')
    d3 = Dependent([[1, 2, 3],[4, 5, 6]], parameters = {'p1': [1, 2], 'p2': [10, 20, 30]})
    print(d3)
    
    print('*** Indexing and slicing ***')
    print(d3[1,2])
    print(d3[0,:])
    print(d3[:,1])
    print(d3[:,0:1])
    
    print('*** Create a complex128 Dependent (default type) ***')
    d4 = Dependent([[1+4j, 2, 3],[4, 5, 6]], parameters = {'p1': [1, 2], 'p2': [10, 20, 30]}, dtype = np.complex128)
    print(d4)
    
    print('*** Aggreagate Dependents into a dict of Dependents ***')
    dataset = {'d3': d3, 'd4': d4}
    print(dataset)
        
    print('*** Save and reload the dict of Dependents (json files) ***')
    Dependent.save_dict_of_dependents(dataset, 'test.json')
    reloaded_dataset = Dependent.load_dict_of_dependents('test.json')
    print(reloaded_dataset)