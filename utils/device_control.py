"""
    This file is part of the implementation of the device control.
    It is used to control the device by keyboard and joystick.
"""

import pygame
import numpy as np
import threading

class KeyboardController:
    def __init__(self, action_queue, shared_data, lock, running_flag, control_frequency=30):
        self.action_queue = action_queue
        self.shared_data = shared_data
        self.lock = lock
        self.running_flag = running_flag
        self.control_frequency = control_frequency # an excessively high control frequency (Hz) will cause multiple movements per single command, while an excessively low frequency will introduce significant latency

        # initialize the velocity parameters
        self.LIN_SPEED = 0.001   # translational velocity
        self.ANG_SPEED = 0.025   # rotational velocity

        # initialize the Pygame keyboard system
        pygame.init()
        pygame.display.set_mode((150, 150))  # must initialize the display to use the keyboard

        print("Keyboard Control Instructions:")
        print("Translation Controls: W/S (Forward/Backward), A/D (Left/Right), Up/Down Arrow Keys (Up/Down)")
        print("Rotation Controls: J/L (Rotation about X-axis), I/K (Rotation about Y-axis), U/O (Rotation about Z-axis)")
        print("Gripper Control: Spacebar (for closing), Ctrl key (for opening)")
        print("Data Collection: Press any movement key to stop data acquisition")
        print("Function Keys: X (Reset the world and clear data); Y (Stop collection and save); ESC (Exit without saving)")
        print("Function Keys: F (Adjust speed: increases XYZ direction by 5x and angular direction by 5x); R (Adjust speed: decreases XYZ direction by 50x and angular direction by 10x)")
        print(f"Current speed: {self.LIN_SPEED} m/s, {self.ANG_SPEED} rad/s")

        # start the thread
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True 
        self.thread.start()

    def _run(self):
        clock = pygame.time.Clock()
        while self.running_flag[0]:
            pygame.event.pump()
            keys = pygame.key.get_pressed()

            if keys[pygame.K_f]:
                self.LIN_SPEED *= 5
                self.ANG_SPEED *= 5
                if self.LIN_SPEED > 0.005:
                    self.LIN_SPEED = 0.005
                    self.ANG_SPEED = 0.05
                print(f"Speed adjustment is set to: {self.LIN_SPEED} m/s, {self.ANG_SPEED} rad/s")

            if keys[pygame.K_r]:
                self.LIN_SPEED /= 5
                self.ANG_SPEED /= 5
                if self.LIN_SPEED < 0.0002:
                    self.LIN_SPEED = 0.00004
                    self.ANG_SPEED = 0.005
                print(f"Speed adjustment is set to: {self.LIN_SPEED} m/s, {self.ANG_SPEED} rad/s")


            # -----------------------------
            # construct the action
            # -----------------------------
            action = np.zeros(7, dtype=np.float32)
            has_input = False

            # translational control
            if keys[pygame.K_s]: action[0] += self.LIN_SPEED; has_input = True
            if keys[pygame.K_w]: action[0] -= self.LIN_SPEED; has_input = True
            if keys[pygame.K_a]: action[1] -= self.LIN_SPEED; has_input = True
            if keys[pygame.K_d]: action[1] += self.LIN_SPEED; has_input = True
            if keys[pygame.K_UP]: action[2] += self.LIN_SPEED; has_input = True
            if keys[pygame.K_DOWN]: action[2] -= self.LIN_SPEED; has_input = True

            # rotational control
            if keys[pygame.K_j]: action[3] += self.ANG_SPEED; has_input = True
            if keys[pygame.K_l]: action[3] -= self.ANG_SPEED; has_input = True
            if keys[pygame.K_i]: action[4] += self.ANG_SPEED; has_input = True
            if keys[pygame.K_k]: action[4] -= self.ANG_SPEED; has_input = True
            if keys[pygame.K_u]: action[5] += self.ANG_SPEED; has_input = True
            if keys[pygame.K_o]: action[5] -= self.ANG_SPEED; has_input = True

            # gripper control
            with self.lock:
                if keys[pygame.K_SPACE]:
                    self.shared_data['gripper_close'] = True
                    has_input = True
                elif keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
                    self.shared_data['gripper_close'] = False
                    has_input = True

                # 功能键
                self.shared_data['x_button'] = keys[pygame.K_x]
                self.shared_data['y_button'] = keys[pygame.K_y]
                self.shared_data['not_saved'] = keys[pygame.K_ESCAPE]

            # construct the gripper action
            action[6] = 1.0 if self.shared_data['gripper_close'] else 0.0

            # put the action into the queue
            if has_input:
                if self.action_queue.full():
                    self.action_queue.get_nowait()  # prevent accumulation
                self.action_queue.put(action.copy().astype(np.float32))

            clock.tick(self.control_frequency)  # control frequency
    
    def stop(self):
        self.running_flag[0] = False
        self.thread.join()
        pygame.quit()

class JoystickController:
    def __init__(self,action_queue, shared_data, lock, running_flag, control_frequency=30):
        # default configuration parameters (can be overridden)
        self.JOY_DEADZONE = 0.12
        self.LIN_SPEED = 0.001   # translational velocity
        self.ANG_SPEED = 0.025   # rotational velocity
        self.control_frequency = control_frequency    # an excessively high control frequency (Hz) will cause multiple movements per single command, while an excessively low frequency will introduce significant latency
        self.action_queue = action_queue
        self.shared_data = shared_data
        self.lock = lock
        self.running_flag = running_flag

        print("Control Instructions for Joystick:")
        print("Left Joystick: Forward/Backward/Left/Right")
        print("Right Joystick: Up/Down for vertical movement; Left/Right for rotation about the X-axis")
        print("D-Pad: Left/Right for rotation about the Y-axis; Up/Down for rotation about the Z-axis")
        print("A Button: Open/Close the gripper")
        print("X Button: Reset the world and clear data")
        print("Y Button: Stop collection and save")
        print("Left Trigger: Exit the program")
        print("Button 6: Increase speed by 5x and angular speed by 5x")
        print("Button 7: Decrease speed by 5x and angular speed by 5x")

        self.joystick = self._init_joystick()


        # start the thread
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()

    def _run(self):
        clock = pygame.time.Clock()

        while self.running_flag[0]:
            pygame.event.pump()

            # ensure the action queue is empty before starting
            self.action_queue.queue.clear()

            if self.joystick.get_button(6):
                self.LIN_SPEED *= 5
                self.ANG_SPEED *= 5
                if self.LIN_SPEED > 0.005:
                    self.LIN_SPEED = 0.005
                    self.ANG_SPEED = 0.05
                print(f"Adjust speed to : {self.LIN_SPEED} m/s, {self.ANG_SPEED} rad/s")

            if self.joystick.get_button(7):
                self.LIN_SPEED /= 5
                self.ANG_SPEED /= 5
                if self.LIN_SPEED < 0.0002:
                    self.LIN_SPEED = 0.00004
                    self.ANG_SPEED = 0.005
                print(f"Adjust speed to : {self.LIN_SPEED} m/s, {self.ANG_SPEED} rad/s")

            # read joystick input
            lx = self.deadzone(self.joystick.get_axis(0))   # left joystick X
            ly = self.deadzone(self.joystick.get_axis(1))   # left joystick Y
            rx = self.deadzone(self.joystick.get_axis(3))   # right joystick X
            ry = self.deadzone(self.joystick.get_axis(4))   # right joystick Y

            dpad = self.joystick.get_hat(0)                 # D-Pad
            dry = dpad[0]                                   # D-Pad left/right(Y-axis)
            drz = dpad[1]                                   # D-Pad up/down(Z-axis）

            a_button = self.joystick.get_button(0)           # A button
            b_button = self.joystick.get_button(1)           # B button
            x_button = self.joystick.get_button(2)           # X button
            y_button = self.joystick.get_button(3)           # Y button
            not_saved = self.joystick.get_button(4)          # not saved button

            # update shared variables
            with self.lock:
                self.shared_data['x_button'] = x_button
                self.shared_data['y_button'] = y_button
                self.shared_data['not_saved'] = not_saved

                # gripper status
                if a_button:
                    self.shared_data['gripper_close'] = False
                elif b_button:
                    self.shared_data['gripper_close'] = True

            # construct the action
            action = np.zeros(7, dtype=np.float32)
            has_input = any(abs(x) > 1e-3 for x in [lx, ly, rx, ry, dry, drz]) or a_button or b_button

            if has_input:
                action[0] = ly * self.LIN_SPEED   # forward/backward
                action[1] = lx * self.LIN_SPEED   # left/right
                action[2] = -ry * self.LIN_SPEED  # up/down
                action[3] = -rx * self.ANG_SPEED  # rotation about X-axis
                action[4] = dry * self.ANG_SPEED  # rotation about Y-axis
                action[5] = drz * self.ANG_SPEED  # rotation about Z-axis
                grip_value = 1.0 if self.shared_data['gripper_close'] else 0.0
                action[6] = grip_value

                # put the action into the queue
                if self.action_queue.full():
                    self.action_queue.get_nowait()  # prevent accumulation
                self.action_queue.put(action.copy().astype(np.float32))

            clock.tick(self.control_frequency)  # control frequency

    # ================== tool functions: deadzone ==================
    def deadzone(self, value):
        """apply a dead zone to eliminate joystick drift"""
        threshold = self.JOY_DEADZONE
        if abs(value) < threshold:
            return 0.0
        sign = 1 if value > 0 else -1
        return sign * (abs(value) - threshold) / (1 - threshold)
    
    # init joystick
    def _init_joystick(self):
        pygame.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No joystick detected")

        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(f"Joystick detected: {joystick.get_name()}")
        return joystick
    
    # stop the thread
    def stop(self):
        self.running_flag[0] = False
        self.thread.join()
        pygame.quit()
    