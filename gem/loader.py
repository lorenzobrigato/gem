import pkgutil
import os
from typing import List
import importlib


class InstanceLoader():

    def __init__(self, module_name):
        super(InstanceLoader, self).__init__()
        
        self.module_name = module_name
        if not(module_name in self.modules_names):
            raise ValueError(f'Module from which loading an instance not expected. Choose in: {self.modules_names}')


    def available_classes(self) -> List:
        """ Finds and returns all classes in the module that contain a given method. """

        LOADING_CONFIG = self.loading_config
        folder_path = LOADING_CONFIG[self.module_name]['path']
        searched_method = LOADING_CONFIG[self.module_name]['searched_method']
        
        base_module_name = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
        classes = []
        folders = [x[0] for x in os.walk(folder_path)]

        for loader, module_name, is_pkg in pkgutil.walk_packages(folders):

            module = loader.find_module(module_name).load_module(module_name)
            atr_names = dir(module)

            for attr_n in atr_names:
                attr = getattr(module, attr_n)

                attr_has_method = searched_method in dir(attr)
                if attr_has_method:

                    attr_def_in_module = module.__name__ == attr.__dict__['__module__']
                    if attr_def_in_module:
                    
                        if getattr(attr, searched_method)() is not NotImplemented:

                            elems_path = os.path.normpath(loader.path).split(os.sep)[::-1]
                            n = []
                            for ep in elems_path:
                                if ep != base_module_name:
                                    n.append(ep)
                                else:
                                    break
                            n = n[::-1]
                            
                            module_fname = base_module_name + '.' + '.'.join(n) + '.' + module_name
                            module_full = importlib.import_module(module_fname)
                            
                            attr_full = getattr(module_full, attr_n)
                            classes.append(attr_full)
                            
        return classes


    def available_instances(self) -> List:

        LOADING_CONFIG = self.loading_config
        classes = self.available_classes()
        instances = []

        for cl in classes:
            searched_method = getattr(cl, LOADING_CONFIG[self.module_name]['searched_method'])
            instances.append(searched_method())
        
        if len(instances) > 0:
            if isinstance(instances[0], list):
                instances = sum(instances, [])

        return instances


    def get_class(self, name: str):
        """ Returns the class given the name of the buildable instance.

        Parameters
        ----------
        name : str
            The name of the instance implemented by a Python class. Note that the name does not correspond to
            the Python class name (i.e., `class.__name__`), but to the name of the instance (e.g., 'rn20' for an architectures' instance). 
            A list of available instances can be obtained from the `available_instances` method.
        """

        LOADING_CONFIG = self.loading_config
        classes = self.available_classes()

        for cl in classes:
            searched_method = getattr(cl, LOADING_CONFIG[self.module_name]['searched_method']) 
            
            if self.name_match(name, searched_method):
                return cl
        
        raise ValueError(f'Unknown instance\'s name: {name}\n Available instances\' names are: {self.available_instances()}')


    def build_instance(self, name: str, *args, **kwargs):
        """ Instantiates the adequate callable instance given its name.

        Parameters
        ----------
        name : str
            The name of the instance implemented by a Python class. Note that the name does not correspond to
            the Python class name (i.e., class.__name__), but to the name of the instance (e.g., rn20 for an architectures' instance). 
            A list of available instances can be obtained from the `available_instances` method.
        *args
            Non-keyword arguments that are needed
            to construct the respective Python class.
        **kwargs
            Remaining keyword arguments (such as `transform`) that are needed
            to construct the respective Python class.
        """

        LOADING_CONFIG = self.loading_config
        classes = self.available_classes()

        for cl in classes:
            searched_method = getattr(cl, LOADING_CONFIG[self.module_name]['searched_method']) 

            if self.name_match(name, searched_method):
                
                if self.module_name == "architectures":
                    instance = cl.build_architecture(name, *args, **kwargs)
                else:
                    instance = cl(*args, **kwargs)

                return instance
        
        raise ValueError(f'Unknown instance\'s name: {name}\n Available instances\' names are: {self.available_instances()}')

    
    def name_match(self, name, method_call) -> bool:
        """ Checks the equality condition between the name and the output of the callable method.

        Parameters
        ----------
        name : str
            The name of the instance implemented by a Python class. Note that the name does not correspond to
            the Python class name (i.e., `class.__name__`), but to the name of the instance (e.g., 'rn20' for an architectures' instance). 
            A list of available instances can be obtained from the `available_instances` method.
        method_call: callable
            A callable method that returns a name/list of instance names depending on the instance searched.
        """

        n = method_call()

        if isinstance(n, str):
            if name == n:
                return True
            else: 
                return False

        elif isinstance(n, list):
            if name in n:
                return True
            else: 
                return False

        elif n is NotImplemented:
            return False

        else:
            raise(TypeError(f"Expected string/list/NotImplemented, found {type(n)} for {n}"))
        

    @property
    def modules_names(self):
        return ['datasets', 'pipelines', 'architectures']


    @property
    def loading_config(self):

        cp = os.path.dirname(os.path.abspath(__file__))
        loading_config = {}

        for mn in self.modules_names:

            loading_config[mn] = {'path' : os.path.join(cp, mn), 'searched_method': 'get_ds_name' if mn == 'datasets' else 
                                                                                    'get_pipe_name' if mn == 'pipelines' else 
                                                                                    'get_arch_names'}

        return loading_config
