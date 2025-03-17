package com.ivan.learning;

import java.util.Map;
import java.util.Scanner;

public class TaylorPredictor {
    private static final Map<Float, Float> functionValues = Map.of(
            -0.15f, 0.6151f,
            -0.10f, 0.6418f,
            -0.05f, 0.6678f,
            0.00f, 0.6931f,
            0.05f, 0.7178f,
            0.10f, 0.7419f,
            0.15f, 0.7654f);

    private static final float delta = 0.05f;
    private static final int MAX_LEVEL = 10;

    private static float numericalDiff(float x, int diffOrder) {
        float xPlusDelta = Math.round((x + delta) * 100) / 100.0f;
        float xMinusDelta = Math.round((x - delta) * 100) / 100.0f;
        if (!functionValues.containsKey(xPlusDelta) || !functionValues.containsKey(xMinusDelta)) {
            throw new IllegalArgumentException("Точка выходит за пределы данных. Требуются: " + xPlusDelta + " и " + xMinusDelta);
        }

        if (diffOrder == 1) {
            return (functionValues.get(xPlusDelta) - functionValues.get(xMinusDelta)) / (2 * delta);
        } else {
            return (numericalDiff(xPlusDelta, diffOrder - 1) - numericalDiff(xMinusDelta, diffOrder - 1)) / (2 * delta);
        }
    }

    public static int factorial(int n) {
        if (n <= 1) {
            return 1;
        } else {
            return n * factorial(n - 1);
        }
    }

    private static float computeTaylor(float x, float x0, int maxLevel) {
        float y = functionValues.get(x0);
        float dx = x - x0;

        for (int n = 1; n <= maxLevel; n++) {
            float derivative = numericalDiff(x0, n);
            float term = (derivative / factorial(n)) * (float) Math.pow(dx, n);
            y += term;
        }
        return y;
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        System.out.print("Введите точку x: ");
        float x = scanner.nextFloat();

        System.out.print("Введите x0: ");
        float x0 = scanner.nextFloat();
        System.out.println();
        scanner.close();

        for (int i = 0; i < MAX_LEVEL; i++) {
            try {
                float y = computeTaylor(x, x0, i);
                System.out.printf("Порядок разложения %d:%n", i);
                System.out.printf("f(%.2f) = %.4f%n", x, y);
            } catch (IllegalArgumentException e) {
                System.out.println("\nОшибка: " + e.getMessage());
                break;
            }
        }
    }
}






// f(0,34) = 0,8506
// f(-0,19) = 0,5932

// f(2) = 1,5930
// f(3) = 2,4178
// f(-1) = 0,0181