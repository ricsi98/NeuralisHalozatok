package com.topdowncar.game.gym;

import com.topdowncar.game.gym.IO;
import com.topdowncar.game.entities.Car;

import static com.topdowncar.game.entities.Car.DRIVE_DIRECTION_BACKWARD;
import static com.topdowncar.game.entities.Car.DRIVE_DIRECTION_FORWARD;
import static com.topdowncar.game.entities.Car.DRIVE_DIRECTION_NONE;
import static com.topdowncar.game.entities.Car.TURN_DIRECTION_LEFT;
import static com.topdowncar.game.entities.Car.TURN_DIRECTION_NONE;
import static com.topdowncar.game.entities.Car.TURN_DIRECTION_RIGHT;

public class CarController {
    private Car car;
    private IO io;
    private static final boolean DEBUG = false;

    public CarController(Car car) {
        this.car = car;
        if (!DEBUG) {
            this.io = IO.getInstance();
        }
    }

    // TODO: implement obs, rew, done
    public void control() {
        if (DEBUG) {
            car.getSensorDistances(4);
            car.setDriveDirection(DRIVE_DIRECTION_FORWARD);
            return;
        }
        // send observation, reward, done
        String out = "";
        for (float f : car.getSensorDistances(8)) {
            out = out + f + " ";
        }
        io.printMessage(out);

        // read action -> act
        // shape: vertical <space> horizontal: 0 0 means stay as it is, 1 -1 means steer right while reversing
        String input = io.readMessage();
        int vertical = Integer.parseInt(input.split(" ")[0]);
        int horizontal = Integer.parseInt(input.replace("\n", "").split(" ")[1]);

        if (vertical == 1) {                                        // UP
            car.setDriveDirection(DRIVE_DIRECTION_FORWARD);
        } else if (vertical == -1) {                                // DOWN
            car.setDriveDirection(DRIVE_DIRECTION_BACKWARD);
        } else {                                                    // NONE
            car.setDriveDirection(DRIVE_DIRECTION_NONE);
        }

        if (horizontal == -1) {                                     // LEFT
            car.setTurnDirection(TURN_DIRECTION_LEFT);
        } else if (horizontal == 1) {                               // RIGHT
            car.setTurnDirection(TURN_DIRECTION_RIGHT);
        } else {                                                    // NONE
            car.setTurnDirection(TURN_DIRECTION_NONE);
        }
    }
}
