class CamExtractor():
    """
        Extracts cam features from the model, this one is for AlexNet1D. May need to be tweaked for other models
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_pos, module in self.model.feature_extractor._modules.items():
            x = module(x)  # Forward
            #print(x)
            if int(module_pos) == self.target_layer:
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        x = self.model.fc3(F.relu(self.model.fc2(F.relu(self.model.fc1(x)))))
        # x = self.model.classifier(x)
        return conv_output, x


class SCORECAM_1dcnn():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.model.eval() #set dropout and batch norm layers to evaluation mode before running inferance
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_slice, target_class=None):
        # Score-CAM
        ## The implementation was derived from https://github.com/tabayashi0117/Score-CAM and modified under MIT license

        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model 
        #extractor = CamExtractor(model, target_layer)
        conv_output, model_output = self.extractor.forward_pass(input_slice)
        #print(model_output)
        if target_class is None:
            target_class = np.argmax(F.softmax(model_output).data.numpy())
        pred = F.softmax(model_output,dim=1)[0]
        print(pred)
        pred_class = pred[target_class].data.numpy()
       
        # Get convolution outputs
        target = conv_output[0]
        target = F.relu(target)
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        #print(len(target))
        test_var_weights = []
        input_response_logit_Xb = F.softmax(self.extractor.forward_pass(input_slice)[1],dim=1)[0][target_class].data.numpy()
        validate_cam_masks = []
        validate_cam_Sc = []
        validate_cam_Al_k = []
        for i in range(len(target)): #iterate over each activation map
            # Unsqueeze
            saliency_map = torch.unsqueeze(torch.unsqueeze(target[i,:],0),0)
            #print("salmap",saliency_map.shape)
            # Upsampling to input size, from output feature size of 4 to 256 in this case
            saliency_map = F.interpolate(saliency_map, size=(256), mode='linear', align_corners=False)
            if saliency_map.max() == saliency_map.min():
                continue
            # Scale between 0-1
            norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
            #print(norm_saliency_map)
            # Get the target score
            X_Hl =input_slice*norm_saliency_map
            validate_cam_masks.append(X_Hl)
            validate_cam_Al_k.append(target[i, :].data.numpy())
            
            activation_map_score = self.extractor.forward_pass(X_Hl)[1]
            w = F.softmax(activation_map_score,dim=1)[0]
            w_class =w[target_class]
            validate_cam_Sc.append(w_class)
            test_var_weights.append(w_class)
            #print("w_class",w_class)
            cam += w_class.data.numpy() * target[i, :].data.numpy() #experimenting with implementation
            #cam += w_class.data.numpy() * saliency_map.data.numpy()

            #print("cam",cam)
        cam_final = transform.resize(cam, (256,))
        cam_final = np.maximum(cam_final, 0)
        # print("target",target)
        # print("cam2",cam_final)
        # if np.min(cam_final) != np.max(cam_final):
        #     cam_final = (cam_final - np.min(cam_final)) / (np.max(cam_final) - np.min(cam_final))  # Normalize between 0-1
        # else:
        #     cam_final = (cam_final - np.min(cam_final)) 

        #validation to original paper, can use either cam_final or cam_validation 
        S_c = [x.detach().numpy()-input_response_logit_Xb for x in validate_cam_Sc]
        #print("s_c", S_c)
        sum_alpha_cks = sum([math.exp(x)for x in S_c])
        alpha_ck = [math.exp(x)/sum_alpha_cks for x in S_c]
        #print("alpha_ck", alpha_ck, sum(alpha_ck))

        cam_validation=np.ones(target.shape[1:], dtype=np.float32)
        #print(validate_cam_Al_k)
        for i in range(len(alpha_ck)):
            feature_cam_part=alpha_ck[i]*validate_cam_Al_k[i]
            #print(feature_cam_part)
            cam_validation += feature_cam_part
            #print(cam_validation)
        cam_validation = transform.resize(cam_validation, (256,))
        cam_validation = np.maximum(cam_validation, 0) #ReLu
        
        return cam_validation,pred_class
    
    def frequency_ablation_cam(self,input_slice, target_class, num_freq_divisions, filter_method='rectangular'):
        '''
        Estimates the importance of particular frequency regions towards the classifcation score for a target class at a target layer
        '''

        #TODO: option to replace notch filtered section with just noise but maintain signal power

        conv_output, model_output = self.extractor.forward_pass(input_slice)
        base_accuracy = F.softmax(model_output,dim=1)[0]
        base_accuracy = base_accuracy[target_class].data.numpy()

        # Combine I and Q channels into a complex number
        #input_slice = input_slice.squeeze(0).transpose(0, 1)
        input_complex = input_slice[0].data.numpy()[0] + 1j*input_slice[0].data.numpy()[1]
        
        # Convert to numpy and apply FFT
        input_fft = fft(input_complex) #default to fft size = len input

        # Length of each frequency division
        division_length = input_fft.shape[0] // num_freq_divisions

        accuracy_weights = []

        for i in range(num_freq_divisions):
            # Create a copy of the frequency domain representation
            modified_fft = np.copy(input_fft)

            # Calculate start and end indices for the current division
            start_idx = i * division_length
            end_idx = start_idx + division_length

            # Apply notch filter (zero out the selected frequency band)
            if filter_method == 'rectangular':
                modified_fft[start_idx:end_idx] = 0
            else:
                print('Filter method: ', filter_method, " not recognized")
                raise

            # Convert back to time domain
            modified_time_domain = ifft(modified_fft)

            # Separate back into I and Q channels
            modified_I = np.real(modified_time_domain)
            modified_Q = np.imag(modified_time_domain)

            # Convert to tensors and expand dimensions as needed for the model
            input_tensor_I = torch.from_numpy(modified_I).float()
            input_tensor_Q = torch.from_numpy(modified_Q).float()
            modified_input_tensor = torch.stack([input_tensor_I, input_tensor_Q], dim=0)
            modified_input_tensor = modified_input_tensor.expand(1, -1, -1)

            # Generate CAM and record accuracy
            score_pred = self.extractor.forward_pass(modified_input_tensor)[1]
            accuracy_weight= F.softmax(score_pred,dim=1)[0][target_class].data.numpy()
            accuracy_weights.append(accuracy_weight)


        #calculate freq cam response as the drop in accuracy due to removal of frequency division
        cam_responses = [abs(base_accuracy - acc) for acc in accuracy_weights ]

        return cam_responses