classdef FigureFactory < Singleton
    %FIGUREFACTORY Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        DefaultPath
        figtypes
        resolution
        ext
    end
    
    methods(Access=private)
        function obj = FigureFactory(defpath)
            %FIGUREFACTORY Construct an instance of this class
            %   Detailed explanation goes here
            if exist('defpath','var')
                if isfolder(defpath)
                    obj.DefaultPath =defpath;
                end
            else
                obj.DefaultPath = pwd;
            end
            obj.ext={'.png','.pdf'};%
            obj.resolution='-r600';
            obj.resolution=300;
        end
    end
    methods (Static)
        function obj = instance(defpath)
            persistent uniqueInstance
            if isempty(uniqueInstance)
                try obj = logistics.FigureFactory(defpath);catch, obj = logistics.FigureFactory();end
                uniqueInstance = obj;
            else
                obj = uniqueInstance;    
                try obj.DefaultPath=defpath;catch, end
            end
        end
    end
    methods
        function [] = saveDepricated(obj,folderfile)
            [filepath,name,~]=fileparts(folderfile);
            if isempty( filepath)||strcmp(filepath,"")
                filepath=obj.DefaultPath;
            end
            if ~isfolder(filepath)
                mkdir(filepath)
            end
            f=gcf;
            try
                f.Renderer='vector';
            catch
                f.Renderer='opengl';
            end
            folderfile=fullfile(filepath,matlab.lang.makeValidName(name));
            for ifig=1:numel(obj.figtypes)
                figtype=obj.figtypes{ifig};
                try
                    print([folderfile obj.ext{ifig}],figtype,obj.resolution,'-bestfit' )
                catch                
                    print([folderfile obj.ext{ifig}],figtype,obj.resolution )
                end
            end
        end
        function [] = save(obj,folderfile)
            [filepath,name,~]=fileparts(folderfile);
            if isempty( filepath)||strcmp(filepath,"")
                filepath=obj.DefaultPath;
            end
            if ~isfolder(filepath)
                mkdir(filepath)
            end
            f=gcf;
            try
                f.Renderer='vector';
            catch
                f.Renderer='opengl';
            end
            folderfile=fullfile(filepath,name);
            for ifig=1:numel(obj.ext)
                    exportgraphics(f,strcat(folderfile, obj.ext{ifig}), ...
                        ContentType="vector",Resolution=obj.resolution ...
                        ,Append=false)
            end
        end
    end
end

