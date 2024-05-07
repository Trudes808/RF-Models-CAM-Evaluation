
#different plotting methods utils
import numpy as np
import matplotlib.pyplot as plt
import cv2



def dataloader_to_complex_arr(trans):
    '''
    Converts a dataset of N tensors with 2 channels (I/Q) of slice len M to a numpy array of NxM complex values per label. Assumes the dataset is ordered.
    Outputs dict of labels, each containing NxM complex IQ samples for that example.
    '''
    #create nnumpy complex
    trans_full_arr = np.array([])
    #print(len(trans))
    for i in range(0,len(trans)):
        np_slice = trans[i,:,:].data.numpy()
        complex_slice = np_slice[0] + 1j*np_slice[1] #0 is I, 1 is Q
        trans_full_arr =np.append(trans_full_arr,complex_slice)
    
    return trans_full_arr

class RF_Visualizer():
    """
        Holds all relevant information for plotting and RF dataset
    """

    def __init__(self, trans,label,sample_rate):
        '''
        trans: tensor of continuous transmission in format (num_slices,channels,slice_size)
        label:tensor of label for that transmission
        '''
        self.trans = trans
        self.sample_rate = sample_rate
        self.label = label
        self.trans_full_arr = dataloader_to_complex_arr(trans)
        

    def plotSpectrogram(self, fftWindow, fftSize,winLen,overlap):
        #adapted from https://github.com/jgibbard/iqtool/blob/master/iqplot.py
        Fs = self.sample_rate
        data = self.trans_full_arr
        #kwargs={'interpolation':'none'}
        if fftSize == None:
            N = len(data)
        else:
            N = fftSize    
        
        if fftWindow == "rectangular":
            spectrum,freqs,t,im =plt.specgram(data, NFFT=N, Fs=Fs, 
            window=lambda data: data*np.ones(len(data)),  noverlap=overlap)
            transpose_spectrum = np.array(spectrum).T
            #print(len(data))
            #print(transpose_spectrum.shape)
            plt.imshow(transpose_spectrum, aspect='auto', extent = [Fs/-2/1e6, Fs/2/1e6, 0, len(data)/Fs], interpolation = 'none',cmap ='gray')

        elif fftWindow == "bartlett":
            plt.specgram(data, NFFT=N, Fs=Fs, 
            window=lambda data: data*np.bartlett(len(data)),  noverlap=int(N/10))
        elif fftWindow == "blackman":
            plt.specgram(data, NFFT=N, Fs=Fs, 
            window=lambda data: data*np.blackman(winLen),  noverlap=overlap)
        elif fftWindow == "hamming":
            plt.specgram(data, NFFT=N, Fs=Fs, 
            window=lambda data: data*np.hamming(len(data)),  noverlap=int(N/10))
        elif fftWindow == "hanning":
            plt.specgram(data, NFFT=N, Fs=Fs, 
            window=lambda data: data*np.hanning(len(data)),  noverlap=int(N/10))

        plt.xlabel("Frequency [MHz]")
        plt.ylabel("Time [s]")
        plt.show()
    
    def plotPSD(self,fftWindow,fftSize):
        #adapted from https://github.com/jgibbard/iqtool/blob/master/iqplot.py
        Fs = self.sample_rate
        data= self.trans_full_arr
        assert fftWindow in ['rectangular', 'bartlett', 'blackman', 
                            'hamming', 'hanning']
        
        if fftSize == None:
            N = len(data)
        else:
            N = fftSize  
        
        #Generate the selected window
        if fftWindow == "rectangular":
            window = np.ones(N)
        elif fftWindow == "bartlett":
            window = np.bartlett(N)
        elif fftWindow == "blackman":
            window = np.blackman(N)
        elif fftWindow == "hamming":
            window = np.hamming(N)
        elif fftWindow == "hanning":
            window = np.hanning(N)         
            
        #dft = np.fft.fft(data*window)    
        
        if Fs == None:
            #If the sample rate is not known then plot PSD as
            #Power/Freq in (dB/Hz)
            plt.psd(data, NFFT=N)
            
        else:
            #If sample rate is known then plot PSD as
            #Power/Freq as (dB/rad/sample)
            plt.psd(data, NFFT=N, Fs=Fs)

        plt.show()


    def plot_spec_with_time_cam_heatmap(self,cams_arr, title, fftWindow='rectangular', fftSize=256):
        Fs = self.sample_rate
        data = self.trans_full_arr
        
        #create spectrogram background from original data
        winLen = fftSize
        overlap = fftSize-1
        if fftWindow == "rectangular":
            spectrum,freqs,t,im =plt.specgram(data, NFFT=fftSize, Fs=Fs, 
            window=lambda data: data*np.ones(winLen),  noverlap=overlap)
            transpose_spectrum = np.array(spectrum).T
            #print(len(data))
            #print(transpose_spectrum.shape)
            #plt.imshow(transpose_spectrum, aspect='auto', extent = [Fs/-2/1e6, Fs/2/1e6, 0, len(data)/Fs], interpolation = 'none',cmap ='gray')

        
        #normalize cam for plotting
        if np.min(cams_arr) != np.max(cams_arr):
            cam_final = (cams_arr - np.min(cams_arr)) / (np.max(cams_arr) - np.min(cams_arr))  # Normalize between 0-1
        else:
            cam_final = (cams_arr - np.min(cams_arr)) 

        # Reshape and normalize cam_arr to create cam_img
        print(cams_arr)
        cam_img = np.tile(cams_arr, (transpose_spectrum.shape[1], 1)).T  # Duplicate each row across W columns
        print(cam_img)
        cam_img = (cam_img * 255).astype(np.uint8)  # Convert to 8-bit format

        #overlay heatmap on time axis according to cam
        heatmap_img = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)

        #prep section of spectrum corresponding to heatmap
        transpose_spectrum_sliced = transpose_spectrum[0:len(heatmap_img),:]
        transpose_spectrum_sliced = cv2.normalize(transpose_spectrum_sliced, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        transpose_spectrum_sliced = np.uint8(transpose_spectrum_sliced)
        transpose_spectrum_sliced = cv2.cvtColor(transpose_spectrum_sliced, cv2.COLOR_GRAY2BGR)
        transpose_spectrum_sliced = transpose_spectrum_sliced.astype(np.uint8)

        print("about to superimpose",heatmap_img.shape,transpose_spectrum_sliced.shape)
        super_imposed_img = cv2.addWeighted(heatmap_img, 0.5, transpose_spectrum_sliced, 0.5, 0)

        #show plot
        plt.imshow(super_imposed_img, aspect='auto', extent = [Fs/-2/1e6, Fs/2/1e6, 0, len(cams_arr)/Fs], interpolation = 'none')
        plt.title(title)
        plt.xlabel("Frequency [MHz]")
        plt.ylabel("Time [s]")
        plt.show()