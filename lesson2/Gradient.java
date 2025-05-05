package com.ivan.learning.lesson2;

import java.util.Random;

public class Gradient {
    private static final float STEP_SIZE = 0.0005f;
    private static final float GRADIENT_THRESHOLD = 0.3f;
    private static final int MAX_ITERATIONS = 100;
    private static final int INITIAL_POINTS = 1000;
    private static float learningRate = 1.0f;

    private static float func(float x, float y) {
        return (float) (Math.pow(0.4 * x, 4) + Math.pow(0.3 * y, 2) + 1.2 * Math.sin(x * y));
    }

    private static float[] computeGradientStep(float currentX, float currentY) {
        float gradientX = (func(currentX + STEP_SIZE, currentY) -
                func(currentX - STEP_SIZE, currentY)) / (2 * STEP_SIZE);
        float gradientY = (func(currentX, currentY + STEP_SIZE) -
                func(currentX, currentY - STEP_SIZE)) / (2 * STEP_SIZE);

        float gradientMagnitude = (float) Math.sqrt(gradientX * gradientX + gradientY * gradientY);
        if (gradientMagnitude < GRADIENT_THRESHOLD) {
            learningRate *= 0.5f;
        }

        float stepX = learningRate * gradientX / gradientMagnitude;
        float stepY = learningRate * gradientY / gradientMagnitude;
        return new float[] {currentX - stepX, currentY - stepY};
    }

    public static void main(String[] args) {
        Random random = new Random();
        float bestX = 0;
        float bestY = 0;
        float minValue = Float.MAX_VALUE;

        for (int point = 0; point < INITIAL_POINTS; point++) {
            float x = random.nextFloat(-1, 1);
            float y = random.nextFloat(-1, 1);

            for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
                float[] nextPoint = computeGradientStep(x, y);
                x = nextPoint[0];
                y = nextPoint[1];
            }

            float currentValue = func(x, y);
            if (currentValue < minValue) {
                bestX = x;
                bestY = y;
                minValue = currentValue;
            }
        }

        System.out.printf("Минимальное значение: %.6f при x = %.6f, y = %.6f%n",
                minValue, bestX, bestY);
    }
}
