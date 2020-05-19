package com.topdowncar.game.gym;


import com.topdowncar.game.entities.Car;
import com.topdowncar.game.entities.Wheel;
import com.badlogic.gdx.math.Vector2;
import com.badlogic.gdx.physics.box2d.Fixture;
import com.badlogic.gdx.physics.box2d.Contact;
import com.badlogic.gdx.physics.box2d.Manifold;
import com.badlogic.gdx.physics.box2d.ContactImpulse;
import com.badlogic.gdx.physics.box2d.ContactListener;

public class RewardModel implements ContactListener {

    private Car car;
    private Vector2 target;
    private Vector2 prevPos;
    private boolean colliding;

    public RewardModel(Car car, Vector2 target) {
        this.car = car;
        this.target = target;
        this.prevPos = car.getBody().getPosition().cpy();
        this.colliding = false;
    }

    private boolean isCarFixture(Fixture f) {
        for (Fixture cf : car.getBody().getFixtureList()) {
            if (cf == f) {
                return true;
            }
        }
        for (Wheel w : car.getWheels()) {
            for (Fixture cf : w.getBody().getFixtureList()) {
                if (cf == f) {
                    return true;
                }
            }
        }
        return false;
    }

    @Override
    public void beginContact(Contact contact) {
        Fixture fixtureA = contact.getFixtureA();
        Fixture fixtureB = contact.getFixtureB();
        boolean aIsCar = isCarFixture(fixtureA);
        boolean bIsCar = isCarFixture(fixtureB);
        if (aIsCar ^ bIsCar) {
            colliding = true;
            return;
        }
    }

    public float getReward() {
        if (colliding) {
            return -1.0f;
        }
        float prevDist= prevPos.dst(target);
        float currDist = car.getBody().getPosition().dst(target);

        prevPos = car.getBody().getPosition().cpy();

        if (currDist < 3.0) {
            return 1.0f;
        }

        if (prevDist > currDist) {
            return 0.05f;
        }
        return -0.05f;
    }

    @Override
    public void endContact(Contact contact) {
        this.colliding = false;
    }

    @Override
    public void preSolve(Contact contact, Manifold oldManifold) {}

    @Override
    public void postSolve(Contact contact, ContactImpulse impulse) {}

}
