//! Tests for the #[hybrid::route] macro

use xybrid_sdk::hybrid;

#[hybrid::route]
fn test_function(input: String) -> String {
    format!("processed: {}", input)
}

#[test]
fn test_macro_compiles() {
    // Test that the macro compiles and the function works
    let result = test_function("test".to_string());
    assert_eq!(result, "processed: test");
}

#[test]
fn test_macro_preserves_functionality() {
    // Verify the macro doesn't break normal function behavior
    #[hybrid::route]
    fn add_one(x: i32) -> i32 {
        x + 1
    }

    assert_eq!(add_one(5), 6);
}
