import numpy as np
import json
import matplotlib.pyplot as plt

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
        string = f'{self.shape} {self.dtype} Dependent with label "{self.label}"\n'
        
        # Pretty print the parameters
        max_key_len = max(len(key) for key in parameters.keys()) if parameters else 0 # Find the maximum key length for alignment
        string += '-' * (max_key_len+1) + '\n'
        
        string += f'{value}\n'
        
        string += '-' * (max_key_len+1) + '\n'
        
        for key, value in parameters.items():
            # Use an f-string with padding
            # :<{max_key_len} means left-align in a field of width max_key_len
            np.set_printoptions(threshold=15, edgeitems=3)
            string += f"{key:<{max_key_len}}: {value}\n"
            np.set_printoptions(threshold=1000, edgeitems=3) # Reset options for good practice
        return string
    
    def _retrieve_index(self, param_name, int_or_set):
        
        def _find_index(array, target_value):
            
            if target_value == 0.0:
                difference_array = np.abs((array - target_value))
            else:
                difference_array = np.abs((array - target_value)/target_value)
            
            # Find the minimum difference value
            # min_difference = difference_array.min()
            
            # Get the minimum value
            min_value = difference_array.min()
            # Get the index of the minimum value
            min_index = difference_array.argmin()

            if min_value <= 1e-4:
                return min_index
            
            else:
                return None
            
            
            # # Find all indices where the difference equals the minimum difference
            # all_indices = np.where(difference_array == min_difference)[0]
            
            # first_index = all_indices[0]
            
            # return first_index
        
        if isinstance(int_or_set, set):
            if len(int_or_set) == 1:
                # Convert the set to a list and return the first (and only) item
                target_value = list(int_or_set)[0]
                index = _find_index(self.parameters[param_name], target_value)
                if index == None:
                    raise Exception(f'Index not found for value {target_value}')
            else:
                raise Exception('{} notation accept only one element')
            
        elif isinstance(int_or_set, int):
            index = int_or_set
        
        return index

    def _to_slices(self, key):
        """Converts all indices in a multi-dimensional key to slices using a generator expression."""
        if isinstance(key, int):
            return slice(key, key + 1)
        
        if isinstance(key, slice):
            return key
        
        # Use a generator to convert each element and create a new tuple
        return tuple(slice(item, item + 1) if isinstance(item, int) else item for item in key)

        
    def __getitem__(self, key):
        parameters_names = list(self.parameters.keys())
        # print(key)
        if not isinstance(key, tuple):
            key = (key,)
        
        nparam = len(key)
        
        new_key = [np.nan]*nparam
        new_parameters = {}
        
        for kparam in range(nparam):
            key_token = key[kparam]
            param_name = parameters_names[kparam]
            
            # indexing
            # my_dependent[3]
            # my_dependent[{4.2}]
            if isinstance(key_token, int) or isinstance(key_token, set):
                start = self._retrieve_index(param_name, key_token)
                stop = start+1
                step = None
            
                new_key[kparam] = slice(start, stop, step)
                
                
            # slicing
            elif isinstance(key_token, slice):
                
                # my_dependent[3::]
                # my_dependent[{4.2}::]
                start = key_token.start
                if isinstance(start, int) or isinstance(start, set):
                    start = self._retrieve_index(param_name, start)
                
                # my_dependent[:7:]
                # my_dependent[:{4.2}:]
                stop = key_token.stop
                if isinstance(stop, int) or isinstance(stop, set):
                    stop = self._retrieve_index(param_name, stop)
                
                # my_dependent[::2]
                step = key_token.step
                if isinstance(step, int) or isinstance(step, set):
                    step = self._retrieve_index(param_name, step)
                    
                new_key[kparam] = slice(start, stop, step)
                    
            # fancy indexing (mixed list of integers and sets)
            # my_dependent[[3,4,6]]
            # my_dependent[[3,{4.2},{6.3}]]
            elif isinstance(key_token, list):
                
                new_key[kparam] = np.array([self._retrieve_index(param_name, item) for item in key_token])
                    
            else:
                raise Exception(f'unknown type for {key_token}')
            
            
            new_parameters[param_name] = self.parameters[param_name][new_key[kparam]]
        
        # value = self.value[self._to_slices(key)]
        # parameters = {param_name: self.parameters[param_name][key[param_index]]for param_index, param_name in enumerate(self.parameters.keys())}

        new_value = self.value[tuple(new_key)]
        return Dependent(new_value, parameters = new_parameters, label = self.label, dtype = self.dtype)
    
    def __setitem__(self, key, value):
       self.value[self._to_slices(key)] = value
        # self.parameters = {param_name: self.parameters[param_name][key[param_index]]for param_index, param_name in enumerate(self.parameters.keys())}
    
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
        
    def plot(self, ax = None):
        if len(self.parameters)==1:
            param1 = next(iter(self.parameters))
            
            if ax is None:
                # Create a new figure and axes
                fig, ax = plt.subplots()
            else:
                # Use the provided axes
                fig = ax.figure
        
            ax = fig.add_subplot(111)
            ax.plot(self.parameters[param1], self.value)
            ax.set_xlabel(param1)
            ax.set_ylabel(self.label)
            
        elif len(self.parameters)==2:
            gene_params = iter(self.parameters)
            param1 = next(gene_params)
            param2 = next(gene_params)
            x = self.parameters[param1]
            y = self.parameters[param2]
            Y, X = np.meshgrid(y, x)
            Z = self.value    # np.sin(np.sqrt(X**2 + Y**2)) / (np.sqrt(X**2 + Y**2)) # A sinc function
            
            # Plot the 3D surface mesh
            if ax is None:
                # Create a new figure and axes
                fig, ax = plt.subplots()
            else:
                # Use the provided axes
                fig = ax.figure
            ax = fig.add_subplot(111, projection='3d')
            surface = ax.plot_surface(X, Y, Z, cmap='viridis')
            # ax.set_title(f"{self.label}")
            ax.set_xlabel(param1)
            ax.set_ylabel(param2)
            ax.set_zlabel(self.label)
            
        else:
            raise Exception(f'plot method is undefined for ({self.shape}) Dependents')
            
        # plt.show()



if __name__ == "__main__":
    
    print('*** Create an empty Dependent) ***')
    d1 = Dependent()
    print(d1)
    
    print('*** Create a nan Dependent (parameters only) ***')
    d2 = Dependent(parameters = {'p1': [1, 2], 'p2': [10, 20, 30]})
    print(d2)
    
    print('*** Create a float64 Dependent (default type) ***')
    d3 = Dependent([[1, 2, 3],[4, 5, 6]], parameters = {'p1': [1, 2], 'p2': [10, 20, 30]}, label='d3')
    print(d3)
    
    print('*** Indexing and slicing ***')
    
    d8 = d3[0,:].squeeze()
    print(d8[0])
    
    
    print(d3[{0},0:2])
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
    

    plt.close('all')    
    d3.plot()
    d3[0,:].squeeze().plot()
    
    
    
    
    
    
    
    

#             % Validate Parameters struct
#             %nbparams = ndims(ContainedArray);
#             fields = fieldnames(args.Parameters);
#             %             if ~(length(fields) == nbparams)
#             %                 error('Wrong number of fields in argument Parameters');
#             %             end
#             for k = 1:length(fields)
#                 indep=args.Parameters.(fields{k});
#                 nbpts = size(ContainedArray,k);
#                 if ~(size(indep) == [1, nbpts])
#                     if (size(indep) == [nbpts, 1])
#                         indep=indep.';
#                     else
#                         error("Parameter " + fields{k} + " must be row vectors with "+ nbpts + " elements");
#                     end
#                 end
#             end



#         function plot(obj)
#             deps = obj.Dependency;
#             if isscalar(deps)
#                 paramname = deps{1};
#                 plot(obj.Parameters.(paramname),obj.value);
#                 xlabel(paramname);
#                 ylabel(obj.Label);
#             elseif length(deps)==2
#                 param1name = deps{1};
#                 param2name = deps{2};
#                 [X,Y] = meshgrid(obj.Parameters.(param1name), obj.Parameters.(param2name));
#                 Z=obj.value.';
#                 meshc(X,Y,Z);
#                 xlabel(param1name);
#                 ylabel(param2name);
#                 zlabel(obj.Label);
#             else
#                 warning('No plot function defined for Dependents with more than 2 parameters')
#             end

#         end

#         function jsonStr = jsonencode(obj, varargin)
#             %s = struct("Name", obj.Name, "Age", obj.Age);

#             if isreal(obj.ContainedArray)
#                 s = struct(...
#                     "Parameters", obj.Parameters, ...
#                     "ContainedArray", obj.ContainedArray, ...
#                     "Label", obj.Label, ...
#                     "Log", obj.Log);
#             else
#                 s = struct(...
#                     "Parameters", obj.Parameters, ...
#                     "ContainedArrayRe", real(obj.ContainedArray), ...
#                     "ContainedArrayIm", imag(obj.ContainedArray), ...
#                     "Label", obj.Label, ...
#                     "Log", obj.Log);
#             end
#             jsonStr = jsonencode(s, varargin{:});
#         end

#         function jsonwrite(obj, filename)
#             arguments
#                 obj
#                 filename string
#             end
#             fid = fopen(filename,'w');
#             fprintf(fid,'%s',obj.jsonencode);
#             fclose(fid);
#         end

#     end

#     methods(Static)

#         function obj = jsondecode(jsonStr, varargin)
#             % arguments
#             %     jsonStr string
#             % end
#             jsonData = jsondecode(jsonStr, varargin{:});
#             obj=Dependent.struct2dependent(jsonData);
#         end

#         function obj=struct2dependent(inputstruct)
#             arguments
#                 inputstruct (1,1) struct
#             end

#             if isfield(inputstruct, "ContainedArray")   %real values
#                 s(1).ContainedArray = inputstruct.ContainedArray;
                
#             elseif (isfield(inputstruct, "ContainedArrayRe") && isfield(inputstruct, "ContainedArrayIm")) %complex values
#                 s(1).ContainedArray = complex(inputstruct.ContainedArrayRe, inputstruct.ContainedArrayIm);
#             end
#             s(1).Parameters = inputstruct.Parameters;
#             s(1).Label = inputstruct.Label;
#             s(1).Log = inputstruct.Log;


#             obj=Dependent(s(1).ContainedArray, ...
#                 Parameters = s(1).Parameters, ...
#                 Label = s(1).Label, ...
#                 Log = s(1).Log);
#         end

#         function obj = jsonread(filename)
#             arguments
#                 filename string
#             end
#             fid = fopen(filename);
#             raw = fread(fid,inf);
#             str = char(raw');
#             fclose(fid);
#             obj = Dependent.jsondecode(str);
#         end

#     end

#     methods (Access=protected)

#         %*** Parenthesis **************************************************
#         function varargout = parenReference(obj, indexOp)
            
#             indices_cellarray = indexOp(1).Indices;

#             newobj = obj.new;
#             newobj.ContainedArray = obj.ContainedArray(indices_cellarray{:}); % obj.ContainedArray = obj.ContainedArray.(indexOp(1));
#             for k=1:length(indices_cellarray)
#                 if isnumeric(indices_cellarray{k}) %test if the indices are an array of numeric (else is the case of ':', where parameters must not be modified)
#                     newobj.Parameters.(obj.Dependency{k})=obj.Parameters.(obj.Dependency{k})(indices_cellarray{k});
#                 end
#             end

#             if isscalar(indexOp)
#                 varargout{1} = newobj;
#             else
#                 [varargout{1:nargout}] = newobj.(indexOp(2:end));
#             end
#         end

#         function obj = parenAssign(obj,indexOp,varargin)

#             % A REPRENDRE !!!!
#             obj.ContainedArray.(indexOp(1))=varargin{1};


#             % Code d'origine!!!
#             %             % Ensure object instance is the first argument of call.
#             %             if isempty(obj)
#             %                 obj = varargin{1};
#             %             end
#             %             if isscalar(indexOp)
#             %                 assert(nargin==3);
#             %                 rhs = varargin{1};
#             %                 obj.ContainedArray.(indexOp) = rhs.ContainedArray;
#             %                 return;
#             %             end
#             %             [obj.(indexOp(2:end))] = varargin{:};
#         end

#         function n = parenListLength(obj,indexOp,ctx)
#             if numel(indexOp) <= 2
#                 n = 1;
#                 return;
#             end
#             containedObj = obj.(indexOp(1:2));
#             n = listLength(containedObj,indexOp(3:end),ctx);
#         end

#         function obj = parenDelete(obj,indexOp)
#             obj.ContainedArray.(indexOp) = [];
#         end

#         %*** Braces *******************************************************
#         function indices_cellarray = values2indices(obj, values_cellarray)
#             indices_cellarray = cell(size(values_cellarray)); %preallocation
#             for k1 = 1:length(values_cellarray)
#                 referencedvals = values_cellarray{k1};
#                 if isnumeric(referencedvals)
#                     %find the indices corresponding to the references values
#                     parametername = obj.Dependency{k1};
#                     parametervals = obj.Parameters.(parametername);
#                     referencedindices=[];
#                     for k2=1:length(referencedvals)
#                         for k3=1:length(parametervals)
#                             if abs(referencedvals(k2)-parametervals(k3))<=10*eps
#                                 referencedindices = [referencedindices, k3];
#                             end
#                         end
#                     end
#                     indices_cellarray{k1}=referencedindices;
#                 else %(case of ':' in referencedvals)
#                     indices_cellarray{k1} = referencedvals;
#                 end
#             end
#         end

#         function varargout = braceReference(obj,indexOp)
#             newobj = obj.new;
#             indices_cellarray = obj.values2indices(indexOp(1).Indices);

#             %Call parenReference code with the found indices
#             if isscalar(indexOp)
#                 varargout{1} = newobj(indices_cellarray{:});
#             else
#                 [varargout{1:nargout}] = newobj(indices_cellarray{:}).(indexOp(2:end)); %NON TESTE
#             end
#         end

#         function obj = braceAssign(obj,indexOp,varargin)
#             indices_cellarray = obj.values2indices(indexOp(1).Indices);

#             %Call parenAssign code with the found indices
#             if isscalar(indexOp)
#                 obj(indices_cellarray{:}) = varargin{1};
#             else
#                 obj(indices_cellarray{:}).(indexOp(2:end)) = varargin{1:nargin};
#             end

#             % TODO
#             %             if isscalar(indexOp)
#             %                 [obj.Arrays.(indexOp)] = varargin{:};
#             %                 return;
#             %             end
#             %             [obj.Arrays.(indexOp)] = varargin{:};
#         end

#         function n = braceListLength(obj,indexOp,ctx)
#             if numel(indexOp) <= 2
#                 n = 1;
#                 return;
#             end
#             containedObj = obj.(indexOp(1:2));
#             n = listLength(containedObj,indexOp(3:end),ctx);
#         end

#     end

#     methods (Access=public)
#         

#         function out=eval(obj, func)
#             % EVAL applies the function fun to each element of the Dependent.
#             newParameters=struct([]);
#             knew=1;
#             for kold=1:size(obj.Dependency,1)
#                 currentPARAM=obj.Parameters.(obj.Dependency{kold})(:).';
#                 newParameters(1).(obj.Dependency{kold})=currentPARAM;
#                 knew=knew+1;
#             end
#             out=Dependent(func(obj.value), "Parameters", newParameters, "Label", "");
#         end

#         function out = sum(obj)
#             out = sum(obj.ContainedArray,"all");
#         end

#         function out = cat(dim,varargin)
#             numCatArrays = nargin-1;
#             newArgs = cell(numCatArrays,1);
#             for ix = 1:numCatArrays
#                 if isa(varargin{ix},'ArrayWithLabel')
#                     newArgs{ix} = varargin{ix}.ContainedArray;
#                 else
#                     newArgs{ix} = varargin{ix};
#                 end
#             end
#             out = ArrayWithLabel(cat(dim,newArgs{:}));
#         end

#         function varargout = size(obj,varargin)
#             %the computation of the size is modified to take singleton
#             %dimensions (now size can be 3x4x1x1)
#             siz = size(obj.ContainedArray,varargin{:});
#             modifiedsiz = ones(1, length(fieldnames(obj.Parameters)));
#             modifiedsiz(1:length(siz))=siz;
#             [varargout{1:nargout}] = modifiedsiz;
#         end

#         function obj = renameParameter(obj,oldparametername,newparametername)
#             arguments
#                 obj 
#                 oldparametername 
#                 newparametername 
#             end
#             paramindex=find(strcmp(obj.Dependency,oldparametername),1);
#             if ~isempty(paramindex)
#                 obj.Parameters = renameStructField(obj.Parameters,oldparametername,newparametername);
#                 obj.Dependency{paramindex}=newparametername;
#             else
#                 error(oldparametername)
#             end
#         end

#         function result=rdivide(DepL, DepR)
#             %   nwR = nwL\nwT     left deembedding (T=T1\T2)
             
#             if isa(DepL, 'Dependent') && isa(DepR, 'Dependent')
#                 newParameters=struct([]);
#                 knew=1;
#                 for kold=1:size(DepL.Dependency,1)
#                     currentPARAM=DepL.Parameters.(DepL.Dependency{kold})(:).';
#                     newParameters(1).(DepL.Dependency{kold})=currentPARAM;
#                     knew=knew+1;
#                 end
#                 result=Dependent(DepL.value./DepR.value, "Parameters", newParameters, "Label", "");

#             else
#                 error('rdivide not defined for these two types of networks');
#             end
#         end
#     end

#     methods (Static, Access=public)
#         function obj = empty(args)
#             % d = Dependent.empty(Parameters = struct("p1", [1, 2], "p2", [10, 20, 30]));
#             arguments
#                 args.Parameters = struct([]);
#                 args.Label = "";
#             end
#             fnames = fieldnames(args.Parameters);
#             n = length(fnames);
#             sz = nan(1,n);
#             for k = 1:n
#                 sz(k) = length(args.Parameters.(fnames{k}));
#             end
            
#             if n==1
#                 array = nan(sz(1),1);
#             else
#                 array = nan(sz);
#             end

#             obj = Dependent(array, Parameters=args.Parameters, Label=args.Label);
#         end
#     end

    
# end