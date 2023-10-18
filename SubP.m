classdef SubP < dagnn.ElementWise
    
    properties
        scale = 2
        opts = {}
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            outputs{1} = vl_nnSubP(inputs{1}, [],'scale', obj.scale,obj.opts{:}) ;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            derInputs{1} =  vl_nnSubP(inputs{1}, derOutputs{1},'scale', obj.scale,obj.opts{:}) ;
            derParams = {} ;
        end
        %     function outputSizes = getOutputSizes(obj, inputSizes)
        %       outputSizes = {} ;
        %     end
        %
        %     function rfs = getReceptiveFields(obj)
        %         rfs = [] ;
        %     end
        
        function obj = SubP(varargin)
            obj.load(varargin) ;
        end
    end
end
