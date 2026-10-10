//! Shared validation for explicitly supplied cloud gateway URLs.
//! The acceptance policy was previously owned by the Flutter Rust binding.

use url::Url;

pub(crate) fn non_empty(value: Option<&str>) -> Option<&str> {
    value.and_then(|value| {
        let trimmed = value.trim();
        (!trimmed.is_empty()).then_some(trimmed)
    })
}

pub(crate) fn validate_gateway_url(gateway_url: &str) -> Result<String, String> {
    // Do not echo the input: even malformed URLs can contain credentials.
    let parsed =
        Url::parse(gateway_url).map_err(|error| format!("Invalid cloud gateway URL: {error}"))?;
    match parsed.scheme() {
        "http" | "https" => {}
        scheme => {
            return Err(format!(
                "Invalid cloud gateway URL: unsupported scheme '{scheme}'"
            ));
        }
    }

    if !parsed.username().is_empty() || parsed.password().is_some() {
        return Err("Invalid cloud gateway URL: credentials are not allowed".into());
    }
    if parsed.query().is_some() || parsed.fragment().is_some() {
        return Err(
            "Invalid cloud gateway URL: query strings and fragments are not allowed".into(),
        );
    }
    let host = parsed
        .host_str()
        .ok_or_else(|| "Invalid cloud gateway URL: host is required".to_string())?;
    if !is_v1_gateway_base(&parsed) {
        return Err("Invalid cloud gateway URL: base URL must include /v1".into());
    }

    if parsed.scheme() == "https" && is_xybrid_gateway_host(host) {
        return Ok(normalize_gateway_url(parsed));
    }

    #[cfg(debug_assertions)]
    {
        if is_debug_gateway_host(host) {
            return Ok(normalize_gateway_url(parsed));
        }
    }

    Err("Invalid cloud gateway URL: release builds only allow HTTPS Xybrid gateway hosts".into())
}

fn normalize_gateway_url(parsed: Url) -> String {
    parsed.as_str().trim_end_matches('/').to_string()
}

fn is_v1_gateway_base(parsed: &Url) -> bool {
    let path = parsed.path().trim_end_matches('/');
    path == "/v1" || path.starts_with("/v1/")
}

fn is_xybrid_gateway_host(host: &str) -> bool {
    let host = host.trim_end_matches('.').to_ascii_lowercase();
    host == "xybrid.dev" || host.ends_with(".xybrid.dev")
}

#[cfg(debug_assertions)]
fn is_debug_gateway_host(host: &str) -> bool {
    let host = host.trim_end_matches('.').to_ascii_lowercase();
    if host == "localhost" || host.ends_with(".localhost") {
        return true;
    }

    match host.parse::<std::net::IpAddr>() {
        Ok(std::net::IpAddr::V4(ip)) => ip.is_loopback() || ip.is_private() || ip.is_link_local(),
        Ok(std::net::IpAddr::V6(ip)) => {
            ip.is_loopback() || is_ipv6_link_local(ip) || is_ipv6_unique_local(ip)
        }
        Err(_) => false,
    }
}

#[cfg(debug_assertions)]
fn is_ipv6_link_local(ip: std::net::Ipv6Addr) -> bool {
    (ip.segments()[0] & 0xffc0) == 0xfe80
}

#[cfg(debug_assertions)]
fn is_ipv6_unique_local(ip: std::net::Ipv6Addr) -> bool {
    (ip.segments()[0] & 0xfe00) == 0xfc00
}

#[cfg(test)]
mod tests {
    use super::validate_gateway_url;

    #[test]
    fn accepts_versioned_xybrid_https_bases() {
        assert_eq!(
            validate_gateway_url("https://api.xybrid.dev/v1/").unwrap(),
            "https://api.xybrid.dev/v1"
        );
        assert_eq!(
            validate_gateway_url("https://XYBRID.DEV/v1/tenant///").unwrap(),
            "https://xybrid.dev/v1/tenant"
        );
    }

    #[test]
    fn rejects_unversioned_paths_and_impostor_hosts() {
        for input in [
            "https://api.xybrid.dev/",
            "https://api.xybrid.dev/v10",
            "https://xybrid.dev.evil.example/v1",
            "http://api.xybrid.dev/v1",
        ] {
            assert!(validate_gateway_url(input).is_err(), "accepted {input}");
        }
    }

    #[test]
    fn rejects_credentials_query_fragments_and_unsupported_schemes() {
        for input in [
            "https://secret@api.xybrid.dev/v1",
            "https://api.xybrid.dev/v1?key=secret",
            "https://api.xybrid.dev/v1#fragment",
            "file:///v1",
            "not-a-url secret",
        ] {
            let error = validate_gateway_url(input).unwrap_err();
            assert!(!error.contains("secret"), "leaked input in {error}");
        }
    }

    #[cfg(debug_assertions)]
    #[test]
    fn debug_accepts_local_gateway_hosts() {
        for input in [
            "http://127.0.0.1:3001/v1",
            "http://localhost:3001/v1",
            "http://192.168.0.2:3001/v1",
        ] {
            assert!(validate_gateway_url(input).is_ok(), "rejected {input}");
        }
    }

    #[cfg(debug_assertions)]
    #[test]
    fn debug_ipv6_behavior_matches_the_previous_flutter_validator() {
        // Url::host_str retains the brackets, so the existing IpAddr check
        // rejects these even though its IPv6 predicates look permissive.
        for input in ["http://[::1]:3001/v1", "http://[fc00::1]:3001/v1"] {
            assert!(validate_gateway_url(input).is_err(), "accepted {input}");
        }
    }

    #[cfg(not(debug_assertions))]
    #[test]
    fn release_rejects_local_gateway_hosts() {
        assert!(validate_gateway_url("http://127.0.0.1:3001/v1").is_err());
    }
}
