package com.topdowncar.game.screens;

import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.Input;
import com.badlogic.gdx.Screen;
import com.badlogic.gdx.graphics.Color;
import com.badlogic.gdx.graphics.GL20;
import com.badlogic.gdx.graphics.OrthographicCamera;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;
import com.badlogic.gdx.graphics.glutils.ShapeRenderer;
import com.badlogic.gdx.graphics.glutils.ShapeRenderer.ShapeType;
import com.badlogic.gdx.physics.box2d.Box2DDebugRenderer;
import com.badlogic.gdx.physics.box2d.World;
import com.badlogic.gdx.utils.viewport.FitViewport;
import com.badlogic.gdx.utils.viewport.Viewport;
import com.topdowncar.game.entities.Car;
import com.topdowncar.game.tools.MapLoader;
import com.topdowncar.game.gym.*;
import com.badlogic.gdx.math.Vector2;

import static com.topdowncar.game.Constants.DEFAULT_ZOOM;
import static com.topdowncar.game.Constants.GRAVITY;
import static com.topdowncar.game.Constants.POSITION_ITERATION;
import static com.topdowncar.game.Constants.PPM;
import static com.topdowncar.game.Constants.RESOLUTION;
import static com.topdowncar.game.Constants.VELOCITY_ITERATION;

public class PlayScreen implements Screen {

    private static final float CAMERA_ZOOM = 0.3f;
    private final SpriteBatch mBatch;
    private final World mWorld;
    private final Box2DDebugRenderer mB2dr;
    private final OrthographicCamera mCamera;
    private final Viewport mViewport;
    private final Car mPlayer;
    private final MapLoader mMapLoader;
    private final CarController carController;
    private Vector2 target;
    private ShapeRenderer shapeRenderer;

    /**
     * Base constructor for PlayScreen
     */
    public PlayScreen() {
        mBatch = new SpriteBatch();
        mWorld = new World(GRAVITY, true);
        mB2dr = new Box2DDebugRenderer();
        mCamera = new OrthographicCamera();
        mCamera.zoom = DEFAULT_ZOOM;
        mViewport = new FitViewport(RESOLUTION.x / PPM, RESOLUTION.y / PPM, mCamera);
        mMapLoader = new MapLoader(mWorld, this);
        mPlayer = new Car(35.0f, 0.3f, 80, mMapLoader, Car.DRIVE_2WD, mWorld);
        RewardModel rm = new RewardModel(mPlayer, this.target);
        mWorld.setContactListener(rm);
        carController = new CarController(mPlayer, target, rm);
        shapeRenderer = new ShapeRenderer();
    }

    @Override
    public void show() {

    }

    public void setTarget(Vector2 target) {
        this.target = target;
    }

    @Override
    public void render(float delta) {
        Gdx.gl.glClearColor(0, 0, 0, 1);
        Gdx.gl.glClear(GL20.GL_COLOR_BUFFER_BIT);
        handleInputKeyboard();
        update(delta);
        draw();
    }

    /**
     * Handling user input and using {@link Car} class to assign direction values
     * Also handling other input, such as escape to quit the game and camera zoom
     */
    private void handleInputKeyboard() {
        if (Gdx.input.isKeyPressed(Input.Keys.Q)) {
            mCamera.zoom -= CAMERA_ZOOM;
        } else if (Gdx.input.isKeyPressed(Input.Keys.E)) {
            mCamera.zoom += CAMERA_ZOOM;
        }

        this.carController.control(mCamera);
    }

    /**
     * Used only for graphic to draw stuff
     */
    private void draw() {
        mBatch.setProjectionMatrix(mCamera.combined);

        // draw target
        shapeRenderer.setProjectionMatrix(mCamera.combined);
        shapeRenderer.begin(ShapeType.Filled);
        shapeRenderer.setColor(Color.RED);
        shapeRenderer.circle(target.x, target.y, 1f);
        shapeRenderer.end();

        mB2dr.render(mWorld, mCamera.combined);
    }

    /**
     * Main update method used for logic
     * @param delta delta time received from {@link PlayScreen#render(float)} method
     */
    private void update(final float delta) {
        mPlayer.update(delta*10);
        mCamera.position.set(mPlayer.getBody().getPosition(), 0);
        mCamera.update();
        mWorld.step(delta*10, VELOCITY_ITERATION, POSITION_ITERATION);
    }

    @Override
    public void resize(int width, int height) {
        mViewport.update(width, height);
    }

    @Override
    public void pause() {

    }

    @Override
    public void resume() {

    }

    @Override
    public void hide() {

    }

    @Override
    public void dispose() {
        mBatch.dispose();
        mWorld.dispose();
        mB2dr.dispose();
        mMapLoader.dispose();
    }
}
