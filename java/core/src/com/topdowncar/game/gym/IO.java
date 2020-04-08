package com.topdowncar.game.gym;

import java.io.*;
import java.util.*;

public class IO {

    private static IO instance = null;
    private Scanner scanner;
    private Writer writer;

    public IO() {
        this.scanner = new Scanner(System.in);
        this.writer = new PrintWriter(System.out);
    }

    public static IO getInstance() {
        if (IO.instance == null) {
            IO.instance = new IO();
        }
        return IO.instance;
    }


    public String readInput() {
        return scanner.nextLine();
    }

    public void printOutput(String output) {
        try {
            this.writer.append(output);
            this.writer.flush();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void dispose() {
        try {
            this.writer.close();
            this.scanner.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}