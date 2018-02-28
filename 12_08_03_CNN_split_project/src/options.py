# -*- coding: utf-8 -*-

"""
Created on Tue Feb 27 17:30:18 2018

@author: Sandalfon
"""
import json

class Options(object):
    
    def __init__(self):
        self.data = {}
        self.run = {}
        self.model = {}

    
    def addRunOptions(self, key, value):
        if key in self.run:
                print('For value '+key+' remplacing value '+ str(self.run[key]) + 
                      ' by ' + str(value))
        self.run[key] = value
        
    def addModelOptions(self, key, value):
        if key in self.model:
                print('For value '+key+' remplacing value '+ str(self.model[key]) + 
                      ' by ' + str(value))
        self.model[key] = value


    def addDataOptions(self, key, value):
        if key in self.data:
                print('For value '+key+' remplacing value '+ str(self.data[key]) + 
                      ' by ' + str(value))
        self.data[key] = value
        
    def addOptions(self, field, key, value):
        if field == 'data':
            self.addDataOptions(key, value)
        elif field == 'run':
            self.addRunOptions(key, value)
        elif field == 'model':
           self.addModelOptions(key, value)
        else:
            print('Field  must be in data, model, run; unknow '+ field)
        
    def getRunOptions(self, key):
        if key not in self.run:
            print('Undefine key ' + key) 
            return None
        return  self.run[key]
    
    
    def getDataOptions(self, key):
        if key not in self.data:
            print('Undefine key ' + key) 
            return None
        return  self.data[key]
    
    def getModelOptions(self, key):
        if key not in self.model:
            print('Undefine key ' + key) 
            return None
        return  self.model[key]
    
    def getOptions(self, field, key):
        if field == 'data':
            return self.getDataOptions(key)
        elif field == 'run':
            return self.getRunOptions(key)
        elif field == 'model':
            return self.getModelOptions(key)
        else:
            print('Field  must be in data, model, run; unknow '+ field)
        return None
    
    def getKeysRunOptions(self):
        return(list(self.run.keys()))
    def getKeysDataOptions(self):
        return(list(self.data.keys()))
    def getKeysModelOptions(self):
        return(list(self.model.keys()))
    def getKeysOptions(self,field):
        if field == 'data':
            return self.getKeysDataOptions()
        elif field == 'run':
            return self.getKeysRunOptions()
        elif field == 'model':
            return self.getKeysModelOptions()
        else:
            print('Field  must be in data, model, run; unknow '+ field)
        return None
    def getKeysAllOptions(self):
        return {'data':self.getKeysDataOptions(),
                'model': self.getKeysModelOptions(),
                'run': self.getKeysRunOptions()}
    
    def __formatOptions(self, d, indent=4, sort=True):
        return json.dumps(d, indent=indent, sort_keys=sort)
         
    def __printOptions(self, d, indent=4, sort=True):
        djson = self._formatOptions(d, indent=indent, sort=sort)
        print(djson)
        
    def printModelOptions(self, indent=4, sort=True):
        self.__printOptions(self.model, indent=4, sort=True)
    def printRunOptions(self, indent=4, sort=True):
        self.__printOptions(self.run, indent=4, sort=True)       
    def printDataOptions(self, indent=4, sort=True):
        self.__printOptions(self.data, indent=4, sort=True)
    def printOptions(self, field, indent=4, sort=True):
         if field == 'data':
            self.printDataOptions(indent=4, sort=True)
         elif field == 'run':
            self.printRunOptions(indent=4, sort=True)
         elif field == 'model':
            self.printModelOptions(indent=4, sort=True)
         else:
            print('Field  must be in data, model, run; unknow '+ field)
         return None
    def printAllOptions(self, indent=4, sort=True):
        self.__printOptions({'data' : self.data, 'model':self.model, 'run':self.run}, indent=4, sort=True)
        
    def saveToJson(self, file, indent=4, sort=True):
        full_dict={'data' : self.data, 'model':self.model, 'run':self.run}
        json_dict = self.__formatOptions(full_dict, indent=indent, sort=sort)
        with open(file, 'w', encoding="utf-8") as outfile:
            outfile.write(str(json_dict))
        
    def loadFromJson(self, file):
        output_json = json.load(open(file))
        if 'model' in output_json:
            self.model = output_json['model']
        if 'run' in output_json:
            self.run = output_json['run']
        if 'data' in output_json:
            self.data = output_json['data']
    
    def getModel(self):
        return self.model
     
    def getData(self):
        return self.data
        
    def getRun(self):
        return self.run