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
    private boolean colliding;

    public RewardModel(Car car, Vector2 target) {
        this.car = car;
        this.target = target;
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
        return 1.0f / car.getBody().getPosition().dst(this.target);
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
