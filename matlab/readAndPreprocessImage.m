function Iout = readAndPreprocessImage(filename)

        I = dicomread(filename);
       

        % Some images may be grayscale. Replicate the image 3 times to
        % create an RGB image.
        %if ismatrix(I)
         %   I = cat(3,I,I,I);
        %end

        % Resize the image as required for the CNN.
        I(I<0)=0;
        Iout = imresize(I, [32 32]);
        Iout = Iout(:,:,1);

        % Note that the aspect ratio is not preserved. In Caltech 101, the
        % object of interest is centered in the image and occupies a
        % majority of the image scene. Therefore, preserving the aspect
        % ratio is not critical. However, for other data sets, it may prove
        % beneficial to preserve the aspect ratio of the original image
        % when resizing.
    end
