package com.topdowncar.game.gym;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.ServerSocket;
import java.net.Socket;

public class IO {

    private static final boolean DEBUG = true;
    private static final int PORT = 34343;
    private static IO instance = null;
    private Socket clientSocket;
    private ServerSocket serverSocket;
    private PrintWriter out;
    private BufferedReader in;

    public IO() {
        try {
            ServerSocket serverSocket =
                    new ServerSocket(PORT);
            Socket clientSocket = serverSocket.accept();
            out = new PrintWriter(clientSocket.getOutputStream(), true);
            in = new BufferedReader(new InputStreamReader(clientSocket.getInputStream()));
        } catch (IOException e) {
            System.out.println("Exception caught when trying to listen on port "
                    + PORT + " or listening for a connection");
            System.out.println(e.getMessage());
        }
    }

    public static IO getInstance() {
        if (instance == null) {
            instance = new IO();
        }
        return instance;
    }

    public void printMessage(String msg) {
        if (DEBUG) {
            System.out.println("Sending message: " + msg);
        }
        out.println(msg);
    }

    public String readMessage() {
        String msg = null;
        try {
            msg = in.readLine();
            if (DEBUG) {
                System.out.println("Got message: " + msg);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return msg;
    }

    public void dispose() {
        try {
            in.close();
            out.close();
            serverSocket.close();
            clientSocket.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
