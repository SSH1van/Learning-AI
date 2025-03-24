package com.ivan.learning.lesson2;

import java.util.*;

public class Gradient {
    private static final float d = 0.0005f;
    private static float p = 1.0f;

    private static float func(float x, float y) {
        return (float) (Math.pow(0.4 * x, 4) + Math.pow(0.3 * y, 2) + 1.2 * Math.sin(x * y));
    }

//    private static float func(float x, float y) {
//        return (float) (Math.pow(0.4 * x, 4) + Math.pow(0.3 * y, 3) + 1.2 * Math.sin(x * y));
//    }

    private static float[] diff(float x, float y) {
        float xx = (func(x + d, y) - func(x - d, y)) / (2 * d);
        float yy = (func(x, y + d) - func(x, y - d)) / (2 * d);

        float q = (float) Math.sqrt(Math.pow(xx, 2) + Math.pow(yy, 2));
        if (q < 0.3) {
            p *= 0.5f;
        }
        return new float[] {x - (p * xx / q), y - (p * yy / q)};
    }

    public static void main(String[] args) {
        Random rand = new Random();
        float x, y;
        float minX = 0, minY = 0, minZ = Float.MAX_VALUE;

        for (int i = 0; i < 1000; i++) {
            x = rand.nextFloat(-1, 1);
            y = rand.nextFloat(-1, 1);
            for (int j = 0; j < 100; j++) {
                float[] result = diff(x, y);
                x = result[0];
                y = result[1];
            }
            if (func(x, y) < minZ) {
                minX = x;
                minY = y;
                minZ = func(x, y);
            }
        }
        System.out.println("Минимальное значение: ");
        System.out.println(func(minX, minY) + " при x и y = " + minX + " " + minY);
    }
}
