# from 3D to 2D | IFFT over FE | i.e. decouble Kx slices 


# Notes: 
    # the original k-space data ( [Reps], [Ch] , [PE2] , [PE1] , [FE] : (70, 32, 72, 180, 224) ) was devided into Four sections to easier handling.
    # Kdat01 is the first section ((0:15, 32, 72, 180, 224) ))



with device:
    # ref_2D=sp.ifft(ref_zf,axes=[-1]) #uncomment to apply IFFT
    # kdat01_2D=np.empty(kdat01.shape,dtype=np.complex64)
    gc.collect()
    if device == sp.Device(0):  # GPU
        torch.cuda.empty_cache()

    for r in range(kdat01.shape[0]-1,-1,-1):
        track_memory(f"before Repetiontion:{r}")
        for c in range(kdat01.shape[1]-1,-1,-1):
            # sl=kdat01[r,c,:,:,:].copy()
            kdat01[r,c,:,:,:]=(sp.ifft(kdat01[r,c,:,:,:],axes=[-1])) #uncomment to apply IFFT
            # del sl
            # kdat01[r,]=kdat01[r,0:c,...]
        
        gc.collect()
    # kdat01=(sp.ifft(kdat01[r,c,:,:,:],axes=[-1]))
    # del kdat01
    # del ref_zf

    # print('ref_2D shape:', ref_2D.shape)
    print('kdat_2D shape:', kdat01.shape)