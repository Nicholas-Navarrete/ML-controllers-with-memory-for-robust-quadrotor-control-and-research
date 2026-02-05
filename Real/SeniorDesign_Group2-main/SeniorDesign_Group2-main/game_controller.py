"""
This game controler library was initially created for use with an Xbox One controller.

If you are adding a class for another type of controller, print out all events so you can see which
ones correspond to which buttons on the controller.

As of right now, I do not know how it will work if two or more controllers are connected at once.
TODO: Allow hotswapping controllers (in case it gets disconnected)
"""

import pygame

"""
Event IDs
These 'should' be the same regardless of the controler used.
print(event) to see object attributes (or look at pygame docs).
"""
AXIS_MOTION = 1536  # Joystick/triggers analog position
HAT_MOTION = 1538   # DPad state
BUTTON_DOWN = 1539  # Button pressed
BUTTON_UP = 1540    # Button released



"""
Base game controller class.
"""
class GameController:
    def __init__(self):
        pygame.init()
        self.controllers = []
        self.clock = pygame.time.Clock()

        # Read in all connected controllers (pygame calls them joysticks, I think for historical reasons).
        # For the sake of clarity, I will call them controllers. Outside of here, joysticks will refer to
        # the two-axis thumbsticks on a controller
        for i in range(pygame.joystick.get_count()):
            self.controllers.append(pygame.joystick.Joystick(i))
            self.controllers[-1].init()
            print("Detected joystick '" + self.controllers[-1].get_name() + "'")

        self._init_inputs()
        
    """Define controller specific inputs and initialize them to the default state."""
    def _init_inputs(self):
        # Define in controller specific subclass
        pass

    """Read inputs from pygame events, and update relavent inupt variables."""
    def _read_inputs(self):
        # Define in controller specific subclass
        pass



class XboxOneController(GameController):
    """Define controller specific inputs and initialize them to the default state."""
    def _init_inputs(self):
        # BUTTONS: 1 = Currently pressed, 0 = Currently released
        self.Button_A = 0
        self.Button_B = 0
        self.Button_X = 0
        self.Button_Y = 0
        self.Button_Select = 0
        self.Button_Start = 0
        self.DPad_Up = 0
        self.DPad_Down = 0
        self.DPad_Left = 0
        self.DPad_Right = 0
        self.Thumb_L = 0
        self.Thumb_R = 0
        self.Bumper_L = 0
        self.Bumper_R = 0
        # TRIGGERS: Analog range -1:1, Fully pressed = 1, Fully released = -1
        self.Trigger_L = -1
        self.Trigger_R = -1
        # JOYSTICKS: Analog range -1:1, right/down = 1, Fully left/up = -1, (x, y)
        self.Joystick_L = [0, 0]
        self.Joystick_R = [0, 0]

    """Read inputs from pygame events, and update relavent inupt variables."""
    def _read_inputs(self):
        for event in pygame.event.get():
            # If axis value has changed (joysticks, analog triggers, ...)
            if event.type == AXIS_MOTION:
                if event.axis == 0:
                    self.Joystick_L[0] = event.value
                elif event.axis == 1:
                    self.Joystick_L[1] = event.value
                elif event.axis == 2:
                    self.Joystick_R[0] = event.value
                elif event.axis == 3:
                    self.Joystick_R[1] = event.value
                elif event.axis == 4:
                    self.Trigger_L = event.value
                elif event.axis == 5:
                    self.Trigger_R = event.value

            # If DPad value has changed
            elif event.type == HAT_MOTION:
                # Horizontal
                if event.value[0] == 0:
                    self.DPad_Left = 0
                    self.DPad_Right = 0
                elif event.value[0] == 1:
                    self.DPad_Left = 0
                    self.DPad_Right = 1
                elif event.value[0] == -1:
                    self.DPad_Left = 1
                    self.DPad_Right = 0
                # Vertical
                if event.value[1] == 0:
                    self.DPad_up = 0
                    self.DPad_down = 0
                elif event.value[1] == 1:
                    self.DPad_up = 1
                    self.DPad_down = 0
                elif event.value[1] == -1:
                    self.DPad_up = 0
                    self.DPad_down = 1

            # If button has been pressed
            elif event.type == BUTTON_DOWN:
                if event.button == 0:
                    self.Button_A = 1
                elif event.button == 1:
                    self.Button_B = 1
                elif event.button == 2:
                    self.Button_X = 1
                elif event.button == 3:
                    self.Button_Y = 1
                elif event.button == 4:
                    self.Bumper_L = 1
                elif event.button == 5:
                    self.Bumper_R = 1
                elif event.button == 6:
                    self.Button_Select = 1
                elif event.button == 7:
                    self.Button_Start = 1
                elif event.button == 8:
                    self.Thumb_L = 1
                elif event.button == 9:
                    self.Thumb_R = 1

            # If button has been released
            elif event.type == BUTTON_UP:
                if event.button == 0:
                    self.Button_A = 0
                elif event.button == 1:
                    self.Button_B = 0
                elif event.button == 2:
                    self.Button_X = 0
                elif event.button == 3:
                    self.Button_Y = 0
                elif event.button == 4:
                    self.Bumper_L = 0
                elif event.button == 5:
                    self.Bumper_R = 0
                elif event.button == 6:
                    self.Button_Select = 0
                elif event.button == 7:
                    self.Button_Start = 0
                elif event.button == 8:
                    self.Thumb_L = 0
                elif event.button == 9:
                    self.Thumb_R = 0

                




if __name__ == "__main__":
    x = XboxOneController()
    while True:
        x.clock.tick(60)
        x._read_inputs()
        print(x.Button_A, x.Button_B, x.Thumb_L, x.Thumb_R, x.Trigger_R)