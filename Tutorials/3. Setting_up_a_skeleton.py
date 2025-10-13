import nimblephysics as nimble
import numpy as np
import time

rajagopal_opensim: nimble.biomechanics.OpenSimFile = nimble.RajagopalHumanBodyModel()
skeleton: nimble.dynamics.Skeleton = rajagopal_opensim.skeleton
right_wrist: nimble.dynamics.Joint = skeleton.getJoint("radius_hand_r")

# Set an arbitrary target location
target: np.ndarray = np.array([0.5, 0.5, 0.5])

# Create a GUI and serve on port 8080
gui = nimble.NimbleGUI()
gui.serve(8080)

# Print Position Index of the degrees of freedom
for i in range(skeleton.getNumDofs()):
  print(f"{i} : {skeleton.getDofByIndex(i).getName()}")
  
# Get an initial guess for positions
positions: np.ndarray = skeleton.getPositions()

for i in range(1000):
  # Render the skeleton to the GUI
  gui.nativeAPI().renderSkeleton(skeleton)

  # Get the world location of the wrist
  wrist_pos: np.ndarray = skeleton.getJointWorldPositions([right_wrist])

  # Draw a red line connecting the wrist to the target
  gui.nativeAPI().createLine(key="wrist_error", points=[wrist_pos, target], color=[1.0, 0.0, 0.0, 1.0])

  # Compute the loss
  error = wrist_pos - target
  loss = np.inner(error, error)
  print("Loss["+str(i)+"]: "+str(loss))

  # Get gradient - we will discuss this more in following sections
  d_loss_d_wrist_pos = 2 * (wrist_pos - target)
  d_wrist_pos_d_joint_angles = skeleton.getJointWorldPositionsJacobianWrtJointPositions([right_wrist])
  d_loss_d_joint_angles = d_wrist_pos_d_joint_angles.T @ d_loss_d_wrist_pos

  # Do not update the pelvis position in the world
  d_loss_d_joint_angles[0:6] = 0.0

  # Update the positions
  positions -= 0.05 * d_loss_d_joint_angles
  skeleton.setPositions(positions)

  time.sleep(0.05)

# Block until the GUI is closed
gui.blockWhileServing()