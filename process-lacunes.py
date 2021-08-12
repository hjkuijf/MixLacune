# -*- coding: utf-8 -*-


import os
import torch
import torchvision
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import SimpleITK as sitk
import glob

import torch.nn as nn
import nibabel as nib

import shutil



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(0)


test_data_path = glob.glob(f'input_data/**/')
for x in range(len(test_data_path)):

    t1_path = glob.glob(test_data_path[x]+'/*T1*')
    t2_path = glob.glob(test_data_path[x]+'/*T2*')
    flair_path = glob.glob(test_data_path[x]+'/*_FLAIR*')

    sub_no = str(t1_path[0])
    sub_no = sub_no.rsplit('/', 1)[-1][0:7]
    

    print("Loading: T1, T2, Flair\n")
    im = sitk.ReadImage(t1_path[0])


    #-------------------Functions------------------------------

    def zscore_normalize(img, mask=None):

        """
        normalize a target image by subtracting the mean of the whole brain
        and dividing by the standard deviation
        Args:
            img (nibabel.nifti1.Nifti1Image): target MR brain image
            mask (nibabel.nifti1.Nifti1Image): brain mask for img
        Returns:
            normalized (nibabel.nifti1.Nifti1Image): img with WM mean at norm_value
        """

        img_data = img.get_fdata()
        if mask is not None and not isinstance(mask, str):
            mask_data = mask.get_fdata()
        elif mask == 'nomask':
            mask_data = img_data == img_data
        else:
            mask_data = img_data > img_data.mean()
        logical_mask = mask_data > 0.  # force the mask to be logical type
        mean = img_data[logical_mask].mean()
        std = img_data[logical_mask].std()
        normalized = nib.Nifti1Image((img_data - mean) / std, img.affine, img.header)
        return normalized
    def read_img(path):
        nib_img = nib.load(path)
        normal = zscore_normalize(nib_img)
        normal =  normal.get_fdata()
        normal = normal.astype(np.float32)
        img_as_tensor = torch.from_numpy(normal)
        img_as_tensor = img_as_tensor.permute(2,1,0)
        img_as_tensor = img_as_tensor.unsqueeze(1)
        return img_as_tensor
    def extract_patches_2d(img,patch_shape,step=[1.0,1.0],batch_first=False):
        patch_H, patch_W = patch_shape[0], patch_shape[1]
        if(img.size(2)<patch_H):
            num_padded_H_Top = (patch_H - img.size(2))//2
            num_padded_H_Bottom = patch_H - img.size(2) - num_padded_H_Top
            padding_H = nn.ConstantPad2d((0,0,num_padded_H_Top,num_padded_H_Bottom),0)
            img = padding_H(img)
        if(img.size(3)<patch_W):
            num_padded_W_Left = (patch_W - img.size(3))//2
            num_padded_W_Right = patch_W - img.size(3) - num_padded_W_Left
            padding_W = nn.ConstantPad2d((num_padded_W_Left,num_padded_W_Right,0,0),0)
            img = padding_W(img)
        step_int = [0,0]
        step_int[0] = int(patch_H*step[0]) if(isinstance(step[0], float)) else step[0]
        step_int[1] = int(patch_W*step[1]) if(isinstance(step[1], float)) else step[1]
        patches_fold_H = img.unfold(2, patch_H, step_int[0])
        if((img.size(2) - patch_H) % step_int[0] != 0):
            patches_fold_H = torch.cat((patches_fold_H,img[:,:,-patch_H:,].permute(0,1,3,2).unsqueeze(2)),dim=2)
        patches_fold_HW = patches_fold_H.unfold(3, patch_W, step_int[1])   
        if((img.size(3) - patch_W) % step_int[1] != 0):
            patches_fold_HW = torch.cat((patches_fold_HW,patches_fold_H[:,:,:,-patch_W:,:].permute(0,1,2,4,3).unsqueeze(3)),dim=3)
        patches = patches_fold_HW.permute(2,3,0,1,4,5)
        patches = patches.reshape(-1,img.size(0),img.size(1),patch_H,patch_W)
        #patches = patches[:,0,:,:,:]
        if(batch_first):
            patches = patches.permute(1,0,2,3,4)
            patches = patches[0,:,:,:,:]
        #patches = patches[0,:,:,:,:]
        return patches
    def reconstruct_from_patches_2d(patches,img_shape,step=[1.0,1.0],batch_first=False):
        patches = patches.unsqueeze(1)
        if(batch_first):
            patches = patches.permute(1,0,2,3,4)
        patch_H, patch_W = patches.size(3), patches.size(4)
        img_size = (patches.size(1), patches.size(2),max(img_shape[0], patch_H), max(img_shape[1], patch_W))
        step_int = [0,0]
        step_int[0] = int(patch_H*step[0]) if(isinstance(step[0], float)) else step[0]
        step_int[1] = int(patch_W*step[1]) if(isinstance(step[1], float)) else step[1]
        nrow, ncol = 1 + (img_size[-2] - patch_H)//step_int[0], 1 + (img_size[-1] - patch_W)//step_int[1]
        r_nrow = nrow + 1 if((img_size[2] - patch_H) % step_int[0] != 0) else nrow
        r_ncol = ncol + 1 if((img_size[3] - patch_W) % step_int[1] != 0) else ncol
        patches = patches.reshape(r_nrow,r_ncol,img_size[0],img_size[1],patch_H,patch_W)
        img = torch.zeros(img_size, device = patches.device)
        overlap_counter = torch.zeros(img_size, device = patches.device)
        for i in range(nrow):
            for j in range(ncol):
                img[:,:,i*step_int[0]:i*step_int[0]+patch_H,j*step_int[1]:j*step_int[1]+patch_W] += patches[i,j,]
                overlap_counter[:,:,i*step_int[0]:i*step_int[0]+patch_H,j*step_int[1]:j*step_int[1]+patch_W] += 1
        if((img_size[2] - patch_H) % step_int[0] != 0):
            for j in range(ncol):
                img[:,:,-patch_H:,j*step_int[1]:j*step_int[1]+patch_W] += patches[-1,j,]
                overlap_counter[:,:,-patch_H:,j*step_int[1]:j*step_int[1]+patch_W] += 1
        if((img_size[3] - patch_W) % step_int[1] != 0):
            for i in range(nrow):
                img[:,:,i*step_int[0]:i*step_int[0]+patch_H,-patch_W:] += patches[i,-1,]
                overlap_counter[:,:,i*step_int[0]:i*step_int[0]+patch_H,-patch_W:] += 1
        if((img_size[2] - patch_H) % step_int[0] != 0 and (img_size[3] - patch_W) % step_int[1] != 0):
            img[:,:,-patch_H:,-patch_W:] += patches[-1,-1,]
            overlap_counter[:,:,-patch_H:,-patch_W:] += 1
        img /= overlap_counter
        if(img_shape[0]<patch_H):
            num_padded_H_Top = (patch_H - img_shape[0])//2
            num_padded_H_Bottom = patch_H - img_shape[0] - num_padded_H_Top
            img = img[:,:,num_padded_H_Top:-num_padded_H_Bottom,]
        if(img_shape[1]<patch_W):
            num_padded_W_Left = (patch_W - img_shape[1])//2
            num_padded_W_Right = patch_W - img_shape[1] - num_padded_W_Left
            img = img[:,:,:,num_padded_W_Left:-num_padded_W_Right]
        return img
    m = nn.Upsample(scale_factor=4, mode='nearest')
    d = nn.Upsample(scale_factor=0.25, mode='nearest')

    #-------------------Load volume------------------------------
    t1 = read_img(t1_path[0])
    t2 = read_img(t2_path[0])
    flair = read_img(flair_path[0])
    
    height = t1.shape[2]
    width = t1.shape[3]
    tensor = torch.cat(( t1,t2,flair),1)
    print("Volume created\n")


    #-------------------Prevalence map------------------------------
    print("Starting the registration\n")


    def register():
        import os

        import elastix
        import imageio

        import elastix
        import numpy as np
        import imageio
        import os
        import SimpleITK as sitk
        def change_parameter(input_path, old_text, new_text, output_path):
            """
            replaces the old_text to the next_text in parameter files

            Parameters
            ----------
            input_path : str
                parameter file path to be changed.
            old_text : str
                old text.
            new_text : str
                new text.
            output_path : str
                changed paramter file path.
            Returns
            -------
            None.
            """
            #check if input_path exists
            if not os.path.exists(input_path):
                print(input_path + ' does not exist.')

            a_file = open(input_path)
            list_of_lines = a_file.readlines()
            for line in range(0,len(list_of_lines)):
                if (list_of_lines[line] == old_text):
                    list_of_lines[line] = new_text

            a_file = open(output_path, 'w')
            a_file.writelines(list_of_lines)
            a_file.close()



        # IMPORTANT: these paths may differ on your system, depending on where
        # Elastix has been installed. Please set accordingly.

        #ELASTIX_PATH = os.path.join('elastix-5.0.1-linux/bin/elastix')
        #TRANSFORMIX_PATH = os.path.join('elastix-5.0.1-linux/bin/transformix')
        ELASTIX_PATH = os.path.join('elastix-5.0.1-linux/bin/elastix')
        TRANSFORMIX_PATH = os.path.join('elastix-5.0.1-linux/bin/transformix')

        if not os.path.exists(ELASTIX_PATH):
            raise IOError('Elastix cannot be found, please set the correct ELASTIX_PATH.')
        if not os.path.exists(TRANSFORMIX_PATH):
            raise IOError('Transformix cannot be found, please set the correct TRANSFORMIX_PATH.')

        # Make a results directory if non exists
        if os.path.exists('results') is False:
            os.mkdir('results')

        # Define the paths to the two images you want to register
        target_dir = os.path.join(t1_path[0])
        moving_dir = os.path.join( 'example_data', 'mni.nii')
        moving_mask_dir = os.path.join('example_data', 'Prevalence_map-csv.nii.gz')
        output_dir='results'


        # Define a new elastix object 'el' with the correct path to elastix
        el = elastix.ElastixInterface(elastix_path=ELASTIX_PATH)

        # Register the moving image to the target image with el →
        el.register(
            fixed_image=target_dir,
            moving_image=moving_dir,
            parameters=[os.path.join( 'example_data', 'affine.txt'), os.path.join('example_data', 'bspline.txt')],
            output_dir=os.path.join('results'))
        # NOTE: two TransformParameters files will come out of this. Check which one to use for transformix. One file calls the other, so only provide one.

        # Find the results
        transform_path = os.path.join(output_dir, 'TransformParameters.1.txt')
        result_path = os.path.join(output_dir, 'result.1.nii')


        param_path=transform_path
        for i in range(len(param_path)):
            old_text = '(FinalBSplineInterpolationOrder 3)\n'
            new_text = '(FinalBSplineInterpolationOrder 0)\n'
            change_parameter(param_path , old_text, new_text, param_path)


        # Feed the directory of the parameters from the registration to a tr → 
        tr = elastix.TransformixInterface(parameters=transform_path,
                                        transformix_path=TRANSFORMIX_PATH)                             
        tr.transform_image(moving_mask_dir, output_dir=r'results')


        # Apply it to the moving prostate segmentation → 
        transformed_image_path = tr.transform_image(moving_mask_dir, output_dir=r'results')

        moving_img_mask = sitk.GetArrayFromImage(sitk.ReadImage(transformed_image_path))
        #print(moving_img_mask)


        img1= sitk.ReadImage('results/result.nii')

        Im = img1
        BinThreshImFilt = sitk.BinaryThresholdImageFilter()
        BinThreshImFilt.SetLowerThreshold(1)
        BinThreshImFilt.SetOutsideValue(0)
        BinThreshImFilt.SetInsideValue(1)
        BinIm = BinThreshImFilt.Execute(Im)

        sitk.WriteImage(BinIm, 'results/prevalence_map.nii.gz')


    register()

    print("Registration done\n")

    map_path = 'results/prevalence_map.nii.gz'
    prev_map_itk = sitk.ReadImage(map_path)
    prev_map_arr = sitk.GetArrayFromImage(prev_map_itk)


    #-------------------Prediction RCNN------------------------------
    model = torch.load('model_RCNN.pt', map_location=device)
    model.to(device)
    print("Model Mask RCNN loaded\n")
    print("Predicting with Mask RCNN......\n")

    # Do prediction on all 64 pacthes == 1 slice 
    def pred_patches(upsample_patch):
        upsample_patch = upsample
        patch_pred = torch.zeros(0,1,256,256)
        for f in range(len(upsample)):
        #for f in range(36):

            one_patch = upsample[f,:,:,:]
            model.eval()
            with torch.no_grad():
                prediction = model([one_patch.to(device)])

            mask = prediction[0]['masks']
            mask = mask.cpu()
            threshold, upper, lower = 0.1, 1, 0
            bmask=np.where(mask>threshold, upper, lower)

            if len(mask) !=0:
                mm0 = bmask[0 ,:,:, :]
                for f in range(len(bmask)):
                    m = bmask[f ,:,:, :]
                    mm0 = mm0 + m 
                    #binarize
                    threshold, upper, lower = 0.1, 1, 0
                    fuse=np.where(mm0>threshold, upper, lower)
                    fuse = torch.from_numpy(fuse)
                    fuse = fuse.unsqueeze(0)
                    #print(fuse.shape)
            elif len(mask) == 0:
                fuse = torch.zeros(1,256,256)
                fuse = fuse.unsqueeze(0)
            patch_pred = torch.cat((patch_pred,fuse),0) 
        downsample = d(patch_pred) 
        vol = reconstruct_from_patches_2d(downsample, [height,width], batch_first=False)
        return vol



    slices  = torch.zeros(0,1,height,width)
    for f in range(len(tensor)):    
        one_slice = tensor[f,:,:,:]
        one_slice = one_slice.unsqueeze(0)
        patches = extract_patches_2d(one_slice, [64,64], batch_first=True)
        m = nn.Upsample(scale_factor=4, mode='nearest')
        upsample = m(patches)
        slice_pred = pred_patches(upsample)
        slices = torch.cat((slices,slice_pred),0)
    print("Prediction done\n")

    foo = slices.squeeze(1)
    it_img = sitk.GetImageFromArray(foo)
    it_img.CopyInformation(im)
    sitk.WriteImage(it_img, 'results/rcnn_pred-script.nii.gz')
    rcnn_pred_itk = it_img
    rcnn_pred_arr = foo




    #-------------------Prediction - map------------------------------
    print("Prediction from Mask RCNN - Prevalence map in progress\n")
    im  = sitk.ReadImage('results/rcnn_pred-script.nii.gz')
    arr = sitk.GetArrayFromImage(im)
    
    im2 = sitk.ReadImage('results/prevalence_map.nii.gz')
    arr2 = sitk.GetArrayFromImage(im2)
    
    #arr = rcnn_pred_arr
    #arr2 = prev_map_arr

    out_arr = arr + arr2 
    out_im = sitk.GetImageFromArray(out_arr)
    out_im.CopyInformation(im)

    Im = out_im
    BinThreshImFilt = sitk.BinaryThresholdImageFilter()
    BinThreshImFilt.SetLowerThreshold(1.1)
    BinThreshImFilt.SetUpperThreshold(2)
    BinThreshImFilt.SetOutsideValue(0)
    BinThreshImFilt.SetInsideValue(1)
    BinIm = BinThreshImFilt.Execute(Im)

    sitk.WriteImage(BinIm, 'results/rcnn_pred-map.nii.gz')

    rcnn_pred_map_itk = BinIm
    rcnn_pred_map_arr = sitk.GetArrayFromImage(rcnn_pred_map_itk)



    #-------------------Prediction UNet ------------------------------
    print("Prediction with Unet\n")
    from torchvision.models import resnext50_32x4d

    class ConvRelu(nn.Module):
        def __init__(self, in_channels, out_channels, kernel, padding):
            super().__init__()

            self.convrelu = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
                nn.ReLU(inplace=True)
            )

        def forward(self, x):
            x = self.convrelu(x)
            return x
    class DecoderBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            
            self.conv1 = ConvRelu(in_channels, in_channels // 4, 1, 0)
            
            self.deconv = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, kernel_size=4,
                                            stride=2, padding=1, output_padding=0)
            
            self.conv2 = ConvRelu(in_channels // 4, out_channels, 1, 0)

        def forward(self, x):
            x = self.conv1(x)
            x = self.deconv(x)
            x = self.conv2(x)

            return x
    class ResNeXtUNet(nn.Module):

        def __init__(self, n_classes):
            super().__init__()
            
            self.base_model = resnext50_32x4d(pretrained=True)
            self.base_layers = list(self.base_model.children())
            filters = [4*64, 4*128, 4*256, 4*512]
            
            # Down
            self.encoder0 = nn.Sequential(*self.base_layers[:3])
            self.encoder1 = nn.Sequential(*self.base_layers[4])
            self.encoder2 = nn.Sequential(*self.base_layers[5])
            self.encoder3 = nn.Sequential(*self.base_layers[6])
            self.encoder4 = nn.Sequential(*self.base_layers[7])

            # Up
            self.decoder4 = DecoderBlock(filters[3], filters[2])
            self.decoder3 = DecoderBlock(filters[2], filters[1])
            self.decoder2 = DecoderBlock(filters[1], filters[0])
            self.decoder1 = DecoderBlock(filters[0], filters[0])

            # Final Classifier
            self.last_conv0 = ConvRelu(256, 128, 3, 1)
            self.last_conv1 = nn.Conv2d(128, n_classes, 3, padding=1)
                        
            
        def forward(self, x):
            # Down
            x = self.encoder0(x)
            e1 = self.encoder1(x)
            e2 = self.encoder2(e1)
            e3 = self.encoder3(e2)
            e4 = self.encoder4(e3)

            # Up + sc
            d4 = self.decoder4(e4) + e3
            d3 = self.decoder3(d4) + e2
            d2 = self.decoder2(d3) + e1
            d1 = self.decoder1(d2)
            #print(d1.shape)

            # final classifier
            out = self.last_conv0(d1)
            out = self.last_conv1(out)
            out = torch.sigmoid(out)
            
            return out

    rx50 = torch.load('model_UNet32.pt', map_location=device)
    rx50.to(device)
    print("Model rx50 loaded\n")
    
    mask_path = sitk.ReadImage('results/rcnn_pred-map.nii.gz')
    mask_img = sitk.GetArrayFromImage(mask_path)
    mask = torch.from_numpy(mask_img)

    
    #mask = torch.from_numpy(rcnn_pred_map_arr)
    mask = mask.unsqueeze(1)
    volume = torch.cat((tensor, mask),1)

    print("Predicting with UNet rx50\n")
    # Do prediction on all 256 pacthes == 1 slice 
    def pred_patches_UNet(patches):
        patches = patches
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(0)

        train_dataloader = DataLoader(patches, batch_size=1, num_workers=0, shuffle=False)
        inp_tensor = torch.zeros(0,1,32,32)

        for i, (data) in enumerate(train_dataloader):
            if data[:,3,:,:].max()==0:
                data = data[:,3,:,:]
                data = data.unsqueeze(0)
                inp_tensor = torch.cat((inp_tensor,data),0)
            # LAcunes are here 
            elif data[:,3,:,:].max()!=0:
                mask = data[:,3,:,:]
                x = data[:,:3,:,:]
                bla2 = x / 255

                pred = rx50(bla2.to(device))
                pred = pred.detach().cpu().numpy()[0,0,:,:]
                pred_tensor = torch.from_numpy(pred)
                pred_tensor = pred_tensor.unsqueeze(0)
                pred_tensor = pred_tensor.unsqueeze(0)

                ## Apply thresholding
                inp_tensor = torch.cat((inp_tensor,pred_tensor),0)
        return inp_tensor



    slices  = torch.zeros(0,1,height,width)
    for f in range(len(volume)):    
        one_slice  = volume[f,:,:,:]
        one_slice = one_slice.unsqueeze(0)
        
        patches = extract_patches_2d(one_slice, [32,32], batch_first=True)
        bla = pred_patches_UNet(patches)
        vol = reconstruct_from_patches_2d(bla, [height,width], batch_first=False)
        slices = torch.cat((slices,vol),0)

    #a = np.array(slices)
    #threshold, upper, lower = 0.7, 1, 0
    #mask=np.where(a>threshold, upper, lower)
    
    foo = slices.squeeze(1)
    #foo = mask.squeeze(1)
    it_img = sitk.GetImageFromArray(foo)
    it_img.CopyInformation(im)
    sitk.WriteImage(it_img, 'results/unet_pred.nii.gz')

    unet_pred_itk = it_img
    unet_pred_arr = foo
    
    

    print("Done\n")


    #-------------------UNet pred - Map ------------------------------
    print("Prediction from UNet - Prevalence map.....\n")
    im  = sitk.ReadImage('results/unet_pred.nii.gz')
    arr = sitk.GetArrayFromImage(im)
    
    #arr = unet_pred_arr
    
    im2 = sitk.ReadImage('results/prevalence_map.nii.gz')
    arr2 = sitk.GetArrayFromImage(im2)

    out_arr = arr + arr2 
    out_im = sitk.GetImageFromArray(out_arr)
    out_im.CopyInformation(im)

    Im = out_im
    BinThreshImFilt = sitk.BinaryThresholdImageFilter()
    BinThreshImFilt.SetLowerThreshold(1.1)
    #BinThreshImFilt.SetUpperThreshold(2)
    BinThreshImFilt.SetOutsideValue(0)
    BinThreshImFilt.SetInsideValue(1)
    BinIm = BinThreshImFilt.Execute(Im)


    end = '/'+ sub_no + '_space-T1_binary_prediction.nii.gz'
    pred_path = os.path.join('output_data'  + end)

    sitk.WriteImage(BinIm, pred_path)
    print("final prediction done \n")
    
    rem_path = ('results')
    shutil.rmtree(rem_path)
    print("results removed \n")

