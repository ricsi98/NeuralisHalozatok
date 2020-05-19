package com.topdowncar.game.gym;


import com.topdowncar.game.entities.Car;
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

    @Override
    public void beginContact(Contact contact) {
        Fixture fixtureA = contact.getFixtureA();
        Fixture fixtureB = contact.getFixtureB();
        for (Fixture f : car.getBody().getFixtureList()) {
            if (f == fixtureA || f == fixtureB) {
                colliding = true;
                return;
            }
        }
    }

    public float getReward() {
        if (colliding) {
            return -1.0f;
        }
        float prevDist= prevPos.dst(target);
        float currDist = car.getBody().getPosition().dst(target);

        prevPos = car.getBody().getPosition().cpy();

        if (prevDist < currDist) {
            return -0.01f;
        }
        return currDist < 3.0f ? 1.0f : 0.01f;
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
