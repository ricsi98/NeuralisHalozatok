package com.topdowncar.game.gym;

import com.badlogic.gdx.graphics.g2d.BitmapFont;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;
import com.badlogic.gdx.math.Vector2;
import com.topdowncar.game.entities.Car;

public class Logger {
    
    private BitmapFont font;

    public Logger() {
        font = new BitmapFont();
    }

    public void log(Car car, Vector2 target, float reward, int horizonta, int vertical) {
        SpriteBatch sb = new SpriteBatch();
        
        sb.begin();
        font.draw(sb, "Car position " + car.getBody().getPosition().x + " " + car.getBody().getPosition().y, 10, 450);
        font.draw(sb, "Target position " + target.x + " " + target.y, 10, 435);
        font.draw(sb, "Distance " + car.getBody().getPosition().dst(target), 10, 420);
        font.draw(sb, "Reward " + reward, 10, 405);
        font.draw(sb, "Action horizontal: " + horizonta + " vertical: " + vertical, 10, 390);
        sb.end();
    }
}