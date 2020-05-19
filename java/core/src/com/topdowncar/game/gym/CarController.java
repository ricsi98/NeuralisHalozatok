package com.topdowncar.game.gym;

import com.topdowncar.game.gym.*;
import com.topdowncar.game.entities.Car;

import com.badlogic.gdx.math.Vector2;


import com.badlogic.gdx.graphics.OrthographicCamera;

import static com.topdowncar.game.entities.Car.DRIVE_DIRECTION_BACKWARD;
import static com.topdowncar.game.entities.Car.DRIVE_DIRECTION_FORWARD;
import static com.topdowncar.game.entities.Car.DRIVE_DIRECTION_NONE;
import static com.topdowncar.game.entities.Car.TURN_DIRECTION_LEFT;
import static com.topdowncar.game.entities.Car.TURN_DIRECTION_NONE;
import static com.topdowncar.game.entities.Car.TURN_DIRECTION_RIGHT;

public class CarController {
    private Car car;
    private IO io;
    private Vector2 target;
    private RewardModel rewardModel;
    private static final boolean DEBUG = false;
    private Logger logger;

    public CarController(Car car, Vector2 target, RewardModel rewardModel) {
        this.car = car;
        this.rewardModel = rewardModel;
        this.target = target;
        this.logger = new Logger();
        if (!DEBUG) {
            this.io = IO.getInstance();
        }
    }

    // TODO: implement obs, rew, done
    public void control(final OrthographicCamera mCamera) {
        if (DEBUG) {
            car.getSensorDistances(4, mCamera);
            car.setDriveDirection(DRIVE_DIRECTION_FORWARD);
            System.out.println("REWARD " + rewardModel.getReward() + " pos " + car.getBody().getPosition() + " target " + target);
            return;
        }

        logger.log(this.car, this.target);

        // send observation, reward, done
        // out size: #sensor + 3
        String out = "";
        for (float f : car.getSensorDistances(16, mCamera)) {
            out = out + f + " ";
        }
        Vector2 dir = target.cpy().sub(car.getBody().getPosition()).nor();
        float angle = car.getBody().getLinearVelocity().angleRad(dir);
        float distance = target.dst(car.getBody().getPosition());
        float speed = car.getBody().getLinearVelocity().len();
        out = out + angle + " " + distance + " " + speed  + " " + rewardModel.getReward() + "\n";
        System.out.println("SENDING THIS:" + out);
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
