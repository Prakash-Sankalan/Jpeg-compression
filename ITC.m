clc; clear; close all;

% Load and preprocess the high-resolution image
imagePath = 'moon.jpg';  % Change this to your image file
originalImage = imread(imagePath);
[height, width, numChannels] = size(originalImage);

% Check if image is RGB, if not convert to RGB
if numChannels == 1
    originalImage = repmat(originalImage, [1 1 3]);
    numChannels = 3;
end

% Define block sizes to test
blockSizes = [8, 16, 32];
compressionRatios = zeros(size(blockSizes));
psnrValues = zeros(size(blockSizes));

% Create figure for results
figure('Position', [100, 100, 1200, 800]);

% Display original image
subplot(3, 4, 1);
imshow(originalImage);
title('Original Image');

for b = 1:length(blockSizes)
    blockSize = blockSizes(b);
    
    % 1. Color Transform: RGB to YCbCr
    ycbcrImage = rgb2ycbcr(originalImage);
    
    % Separate Y, Cb, Cr components
    Y = double(ycbcrImage(:,:,1));
    Cb = double(ycbcrImage(:,:,2));
    Cr = double(ycbcrImage(:,:,3));
    
    % 2. Chroma Downsampling (4:2:0 format)
    downsampleFactor = 2;
    Cb_downsampled = imresize(Cb, 1/downsampleFactor, 'bicubic');
    Cr_downsampled = imresize(Cr, 1/downsampleFactor, 'bicubic');
    
    % Standard JPEG quantization matrices
    Q_Y = [
        16 11 10 16 24 40 51 61;
        12 12 14 19 26 58 60 55;
        14 13 16 24 40 57 69 56;
        14 17 22 29 51 87 80 62;
        18 22 37 56 68 109 103 77;
        24 35 55 64 81 104 113 92;
        49 64 78 87 103 121 120 101;
        72 92 95 98 112 100 103 99
    ];
    
    Q_C = [
        17 18 24 47 99 99 99 99;
        18 21 26 66 99 99 99 99;
        24 26 56 99 99 99 99 99;
        47 66 99 99 99 99 99 99;
        99 99 99 99 99 99 99 99;
        99 99 99 99 99 99 99 99;
        99 99 99 99 99 99 99 99;
        99 99 99 99 99 99 99 99
    ];
    
    % Resize quantization matrices if block size is not 8
    if blockSize ~= 8
        Q_Y = imresize(Q_Y, [blockSize blockSize], 'nearest');
        Q_C = imresize(Q_C, [blockSize blockSize], 'nearest');
    end
    
    % Pad components to be multiple of blockSize
    Y_padded = pad_to_blocksize(Y, blockSize);
    Cb_padded = pad_to_blocksize(Cb_downsampled, blockSize);
    Cr_padded = pad_to_blocksize(Cr_downsampled, blockSize);
    
    % 3. Forward DCT and 4. Quantization
    [Y_quantized, Y_dct] = apply_dct_quantization(Y_padded, Q_Y, blockSize);
    [Cb_quantized, Cb_dct] = apply_dct_quantization(Cb_padded, Q_C, blockSize);
    [Cr_quantized, Cr_dct] = apply_dct_quantization(Cr_padded, Q_C, blockSize);
    
    % --- Display the DCT matrix (first block of Y component) ---
    fprintf('DCT matrix for Y component (first block) for block size %d:\n', blockSize);
    disp(Y_dct(1:blockSize, 1:blockSize));
    
    % 5. Huffman Encoding
    [Y_encoded, Y_dict] = huffman_encode(Y_quantized);
    [Cb_encoded, Cb_dict] = huffman_encode(Cb_quantized);
    [Cr_encoded, Cr_dict] = huffman_encode(Cr_quantized);
    
    % Calculate compression ratio
    original_size = numel(originalImage) * 8; % 8 bits per pixel component
    compressed_size = length(Y_encoded) + length(Cb_encoded) + length(Cr_encoded);
    compressionRatios(b) = original_size / compressed_size;
    
    % 6. Huffman Decoding
    Y_decoded = huffman_decode(Y_encoded, Y_dict, size(Y_quantized));
    Cb_decoded = huffman_decode(Cb_encoded, Cb_dict, size(Cb_quantized));
    Cr_decoded = huffman_decode(Cr_encoded, Cr_dict, size(Cr_quantized));
    
    % 7. Dequantization and 8. Inverse DCT
    Y_reconstructed = apply_idct_dequantization(Y_decoded, Q_Y, blockSize);
    Cb_reconstructed = apply_idct_dequantization(Cb_decoded, Q_C, blockSize);
    Cr_reconstructed = apply_idct_dequantization(Cr_decoded, Q_C, blockSize);
    
    % 9. Chroma Upsampling
    Cb_upsampled = imresize(Cb_reconstructed, size(Y_reconstructed), 'bicubic');
    Cr_upsampled = imresize(Cr_reconstructed, size(Y_reconstructed), 'bicubic');
    
    % Crop to original dimensions
    Y_final = Y_reconstructed(1:height, 1:width);
    Cb_final = Cb_upsampled(1:height, 1:width);
    Cr_final = Cr_upsampled(1:height, 1:width);
    
    % 10. Color Transform: YCbCr to RGB
    reconstructedYCbCr = zeros(height, width, 3);
    reconstructedYCbCr(:,:,1) = Y_final;
    reconstructedYCbCr(:,:,2) = Cb_final;
    reconstructedYCbCr(:,:,3) = Cr_final;
    reconstructedYCbCr = uint8(reconstructedYCbCr);
    reconstructedRGB = ycbcr2rgb(reconstructedYCbCr);
    
    % Calculate PSNR
    psnrValues(b) = psnr(reconstructedRGB, originalImage);
    
    % Display compressed representation (same for all block sizes)
    subplot(3, 4, 2);
    compressed_display = randn(256, 256); % Random noise to represent compressed data
    imshow(compressed_display, []);
    title('Compressed JPEG Data');
    
    % Display results for this block size
    subplot(3, 4, b+4);
    imshow(reconstructedRGB);
    title(sprintf('Reconstructed (Block %d)', blockSize));
    
    % Display Y component quantized values
    subplot(3, 4, b+8);
    imshow(uint8(Y_quantized), []);
    title(sprintf('Quantized Y (Block %d)', blockSize));
end

% Display compression results
figure;
subplot(1, 2, 1);
bar(blockSizes, compressionRatios);
title('Compression Ratio vs Block Size');
xlabel('Block Size');
ylabel('Compression Ratio');
grid on;

subplot(1, 2, 2);
bar(blockSizes, psnrValues);
title('PSNR vs Block Size');
xlabel('Block Size');
ylabel('PSNR (dB)');
grid on;

fprintf('Compression Results:\n');
fprintf('Block Size\tCompression Ratio\tPSNR (dB)\n');
for b = 1:length(blockSizes)
    fprintf('%d\t\t%.2f:1\t\t\t%.2f\n', blockSizes(b), compressionRatios(b), psnrValues(b));
end

%% Helper Functions

% Pad image to be multiple of blockSize
function padded = pad_to_blocksize(img, blockSize)
    [height, width] = size(img);
    padded_height = ceil(height / blockSize) * blockSize;
    padded_width = ceil(width / blockSize) * blockSize;
    padded = zeros(padded_height, padded_width);
    padded(1:height, 1:width) = img;
end

% Apply DCT and Quantization
function [quantized, dct_coeffs] = apply_dct_quantization(img, Q, blockSize)
    [height, width] = size(img);
    dct_coeffs = zeros(height, width);
    quantized = zeros(height, width);
    
    for i = 1:blockSize:height
        for j = 1:blockSize:width
            block = img(i:i+blockSize-1, j:j+blockSize-1);
            dct_block = dct2(block);
            dct_coeffs(i:i+blockSize-1, j:j+blockSize-1) = dct_block;
            quantized(i:i+blockSize-1, j:j+blockSize-1) = round(dct_block ./ Q);
        end
    end
end

% Apply Inverse DCT and Dequantization
function reconstructed = apply_idct_dequantization(quantized, Q, blockSize)
    [height, width] = size(quantized);
    reconstructed = zeros(height, width);
    
    for i = 1:blockSize:height
        for j = 1:blockSize:width
            block = quantized(i:i+blockSize-1, j:j+blockSize-1);
            dequantized_block = block .* Q;
            reconstructed(i:i+blockSize-1, j:j+blockSize-1) = idct2(dequantized_block);
        end
    end
end

% Huffman Encoding
function [encoded, dict] = huffman_encode(data)
    try
        % Try using Communications Toolbox functions
        symbols = unique(data(:));  % Get unique symbols
        counts = histc(data(:), symbols);  % Get counts of each symbol
        probabilities = counts / sum(counts);  % Normalize to get probabilities

        % Remove zero-probability symbols
        validIdx = probabilities > 0;
        symbols = symbols(validIdx);
        probabilities = probabilities(validIdx);

        % Generate Huffman Dictionary
        dict = huffmandict(symbols, probabilities);
        
        % Encode the image
        encoded = huffmanenco(data(:), dict);
    catch
        % Alternative implementation without Communications Toolbox
        % Simple run-length encoding as fallback
        data_vec = data(:);
        [runValues, runLengths] = rle_encode(data_vec);
        encoded = [runValues; runLengths];
        dict = struct('type', 'rle');  % Mark that we used RLE instead
    end
end

% Huffman Decoding
function decoded = huffman_decode(encoded, dict, originalSize)
    try
        % Try using Communications Toolbox function
        decoded_vector = huffmandeco(encoded, dict); % Decode Huffman encoded data
        decoded = reshape(decoded_vector, originalSize); % Reshape to original size
    catch
        % Alternative implementation without Communications Toolbox
        if isstruct(dict) && strcmp(dict.type, 'rle')
            % If we used RLE encoding as fallback
            runValues = encoded(1:end/2);
            runLengths = encoded(end/2+1:end);
            decoded_vector = rle_decode(runValues, runLengths);
            decoded = reshape(decoded_vector, originalSize);
        else
            error('Huffman decoding failed and no fallback available');
        end
    end
end

% Simple Run-Length Encoding (fallback if huffmanenco is unavailable)
function [values, lengths] = rle_encode(data)
    if isempty(data)
        values = [];
        lengths = [];
        return;
    end
    
    % Find the locations where the data changes
    changes = find([1; diff(data(:)) ~= 0]);
    
    % Calculate run lengths
    lengths = diff([changes; length(data)+1]);
    
    % Get the values
    values = data(changes);
end

% Simple Run-Length Decoding (fallback if huffmandeco is unavailable)
function decoded = rle_decode(values, lengths)
    totalLength = sum(lengths);
    decoded = zeros(totalLength, 1);
    
    idx = 1;
    for i = 1:length(values)
        decoded(idx:idx+lengths(i)-1) = values(i);
        idx = idx + lengths(i);
    end
end
