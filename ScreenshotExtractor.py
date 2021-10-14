def screenshots():
    # Importing all necessary libraries 
    import cv2
    import os
      
    # Read the video from specified path 
    cam = cv2.VideoCapture("E:/Sem 5/PROJECTS/Data Analytics Project/sample_video.mp4")
    
    try: 
        # creating a folder named data 
        if not os.path.exists('data'): 
            os.makedirs('data') 
      
    # if not created then raise error 
    except OSError: 
        print ('Error: Creating directory of data') 
      
    # frame
    currentframe = 0
    
    while(True):
          
        cam.set(cv2.CAP_PROP_POS_FRAMES, currentframe)
        # reading from frame 
        ret,frame = cam.read() 
      
        if ret: 
            # if video is still left continue creating images 
            name = './data/frame' + str(currentframe) + '.jpg'
            print ('Creating...' + name) 
      
            # writing the extracted images 
            cv2.imwrite(name, frame) 
        else: 
            break
        
        currentframe+=90
    # Release all space and windows once done 
    cam.release() 
    cv2.destroyAllWindows()
