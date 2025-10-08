import nimblephysics as nimble
import numpy as np
import time
import torch

# Helper function to convert from a standard quaternion [w, x, y, z]
# to an angle-axis representation [rx, ry, rz] that Nimble uses for FreeJoints.
def quaternion_to_angle_axis(quat):
    """Converts a quaternion to an angle-axis vector."""
    w, x, y, z = quat
    
    # If the quaternion is close to the identity, no rotation is needed.
    if w > 0.99999:
        return np.zeros(3)
        
    angle = 2 * np.arccos(w)
    s = np.sqrt(1 - w*w)
    
    # Avoid division by zero
    if s < 0.00001:
        return np.array([x, y, z]) * angle
        
    axis = np.array([x, y, z]) / s
    return axis * angle

# ====================================================================
# >> YOUR CODE GOES HERE <<
# ====================================================================
def get_orientation():
    """
    *** This is the placeholder function for your IMU data. ***
    
    Replace the contents of this function with the code that reads
    from your actual IMU sensor.
    
    It must return a NumPy array representing a quaternion [w, x, y, z].
    
    For this example, it generates a smooth, continuous rotation based on time.
    """
    t = time.time()
    
    # Create a rotation that wobbles around the x and z axes
    angle = t * 1.5  # Speed of rotation
    axis = np.array([np.sin(t*0.5), 0, np.cos(t*0.5)])
    axis /= np.linalg.norm(axis) # Normalize axis vector
    
    # Convert from axis-angle to quaternion for the return value
    w = np.cos(angle / 2)
    x, y, z = axis * np.sin(angle / 2)
    
    return np.array([w, x, y, z])
# ====================================================================

def customLoopStates(states, gui, box, boxBody):
    for state in states:
        
        
        gui.nativeAPI().renderSkeleton(box)  
        

        
        
        
        gui.nativeAPI().renderBasis(scale=0.3, pos=box_pos, euler=box_euler_angles) 
        
        

# Main execution block
if __name__ == "__main__":

    # 1. Set up the simulation world
    world = nimble.simulation.World()
    world.setGravity([0, 0, 0]) # No gravity for this simple viz

    # 2. Create the object that will represent the IMU
    box = nimble.dynamics.Skeleton()
    boxJoint, boxBody = box.createBallJointAndBodyNodePair()
    boxShape = boxBody.createShapeNode(nimble.dynamics.BoxShape([.1, .1, .05]))
    boxVisual = boxShape.createVisualAspect()
    boxVisual.setColor([0.5, 0.5, 0.5])
    
    world.addSkeleton(box)
    box.setName("IMU_Object")
    
    # # Set up initial conditions for optimization
    initial_rotation: torch.Tensor = torch.tensor([0.0, 0.0, 0.0])
    initial_velocity: torch.Tensor = torch.zeros((world.getNumDofs()), requires_grad=True)
    
    # Set up the GUI
    gui: nimble.NimbleGUI = nimble.NimbleGUI(world)
    gui.serve(8080)
    
    # # Render the skeleton to the GUIright_hand
    gui.nativeAPI().renderSkeleton(box)  
    
    # Convert the right hand rotation matrix to euler angles
    box_world_transform: nimble.math.Isometry3 = boxBody.getWorldTransform()
    
    box_pos = box_world_transform.translation()
    box_euler_angles = nimble.math.matrixToEulerXYZ(box_world_transform.rotation())
    
    # box_euler_angles[1] = 45*3.14/180
    
    # gui.nativeAPI().renderBasis(scale=0.3, pos=box_pos, euler=box_euler_angles, prefix="IMU_basis")   
    # gui.blockWhileServing()
            
    while True:
        
        state: torch.Tensor = torch.cat((initial_rotation, initial_velocity), 0)
        states = [state]
        
        rotation = initial_rotation

        for i in range(90):
            angle_y = i * 3.14 / 180       
            # box_euler_angles = nimble.math.matrixToEulerXYZ(box_world_transform.rotation())
            # box_euler_angles[1] = angle_y
            rotation[1] = angle_y


            # Step the world using Nimble's differentiable timestep
            state = torch.cat((rotation, initial_velocity), 0)
            state = nimble.timestep(world, state, torch.zeros((world.getNumDofs())))

            # Get current transform of the box for rendering basis
            box_world_transform = world.getSkeleton(0).getBodyNode(0).getWorldTransform()
            box_pos = box_world_transform.translation()
            box_euler_angles = nimble.math.matrixToEulerXYZ(box_world_transform.rotation())
            # box_euler_angles[1] = angle_y
            print(box_pos)
            print(box_euler_angles)

            # Render the box and its basis
            gui.nativeAPI().renderWorld(world)
            gui.nativeAPI().renderBasis(scale=0.3, pos=box_pos, euler=box_euler_angles, prefix="IMU_basis", layer="IMU_layer")

            # Refresh GUI
            gui.step()  # This is the correct way to update NimbleGUI
            time.sleep(0.01)
            

         
        
