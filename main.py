import tools
import cv2
import os
import numpy as np

if __name__ == '__main__':

    camerasfile  = 'cameras.xml'
    pointcloud   = 'points.las'
    imagesfolder = 'images'

    # Load the cameras file and bring everything into local ENU coordiante system in meters.
    cameras = tools.CamerasXML().read(camerasfile)

    # Choose a random camera from the scene
    camera = cameras.cameras[171]

    # Camera extrinsics etc ..
    print(f'position={camera.project.position()}')
    print(f'orientation={camera.project.orientation()}')
    # Only use cameras that have been accurately structured
    print(f'orientation={camera.structured}')

    # load the image seen from the camera
    cameraview = cv2.imread(f'{imagesfolder}/{camera.label}')

    # Read the pointcloud and convert from WGS84 to ENU
    points = tools.read_pointcloud(camerasfile, pointcloud)

    # blank image to project points into
    blank = np.zeros(cameraview.shape, dtype='uint8')

    # Project every point into the camera
    print("Projecting some points into a camera - this will be slow")
    for index, point in enumerate(points):

        if index % 100 != 0:
            continue

        # Point color
        r, g, b = point[-3:] * 255.

        # Project to pixel coordinates taking into account distortion
        x, y = camera.project.to_image(point[:3])

        x = int(x[0, 0] + 0.5)
        y = int(y[0, 0] + 0.5)

        # Check pixel falls within image
        if x >= 0 and y >= 0 and x < blank.shape[1] and y < blank.shape[0]:
            cv2.circle(blank, (x, y), 5, [b, g, r])

    # Save the original image and projected pointcloud side-by-side
    print('saving result.png')
    cv2.imwrite("result.png", np.hstack([cameraview, blank]))
