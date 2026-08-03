package ai.xybrid.smoke

/**
 * Compile-time proof the AAR's classes resolve from a consumer: references to
 * the public surface force kotlinc + d8 to load them from classes.jar.
 */
object Smoke {
    val entrypoints = listOf(
        ai.xybrid.Xybrid::class,
        ai.xybrid.Envelope::class,
        ai.xybrid.XybridEnvelope::class,
    )
}
