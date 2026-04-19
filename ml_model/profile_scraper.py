"""
=============================================================
  Instagram Profile Scraper
=============================================================
Fetches public Instagram profile data using instaloader.
Extracts features needed for the fake profile ML model.
"""

import re
import instaloader
import requests


def extract_username_from_url(url: str) -> str:
    """
    Extract Instagram username from various URL formats.

    Supports:
      - https://www.instagram.com/username/
      - https://instagram.com/username
      - http://instagram.com/username/?hl=en
      - instagram.com/username
      - Just a username string
    """
    url = url.strip().rstrip("/")

    # Match Instagram URL patterns
    patterns = [
        r"(?:https?://)?(?:www\.)?instagram\.com/([A-Za-z0-9._]+)",
        r"^([A-Za-z0-9._]+)$",  # Plain username
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            username = match.group(1)
            # Filter out known Instagram paths that aren't usernames
            non_user_paths = {
                "p", "explore", "reels", "stories", "tv",
                "accounts", "directory", "about", "developer",
            }
            if username.lower() not in non_user_paths:
                return username

    return ""


def scrape_instagram_profile(url: str) -> dict:
    """
    Scrape public Instagram profile data.

    Args:
        url: Instagram profile URL or username

    Returns:
        dict with profile data or error info
    """
    username = extract_username_from_url(url)

    if not username:
        return {
            "success": False,
            "error": "Could not extract a valid username from the URL. "
                     "Please enter a valid Instagram profile URL "
                     "(e.g. https://instagram.com/username).",
        }

    try:
        # 1. Try Instagram's web_profile_info API endpoint
        api_url = f"https://www.instagram.com/api/v1/users/web_profile_info/?username={username}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "x-ig-app-id": "936619743392459",
            "Accept": "*/*",
        }
        
        response = requests.get(api_url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            user_data = data.get("data", {}).get("user", {})
            
            if user_data:
                fullname = user_data.get("full_name") or ""
                bio = user_data.get("biography") or ""
                num_followers = user_data.get("edge_followed_by", {}).get("count", 0)
                num_following = user_data.get("edge_follow", {}).get("count", 0)
                num_posts = user_data.get("edge_owner_to_timeline_media", {}).get("count", 0)
                
                profile_pic_url = user_data.get("profile_pic_url", "")
                has_profile_pic = bool(profile_pic_url and "44884218_345707102882519_2446069589734326272_n" not in profile_pic_url)
                
                external_url = bool(user_data.get("external_url"))
                is_private = bool(user_data.get("is_private"))
                is_verified = bool(user_data.get("is_verified"))
                
                return {
                    "success": True,
                    "username": username,
                    "fullname": fullname,
                    "bio": bio,
                    "num_followers": num_followers,
                    "num_following": num_following,
                    "num_posts": num_posts,
                    "profile_pic": has_profile_pic,
                    "external_url": external_url,
                    "private": is_private,
                    "desc_length": len(bio),
                    "profile_pic_url": profile_pic_url,
                    "is_verified": is_verified,
                }
                
        elif response.status_code == 404:
            return {
                "success": False,
                "error": f"Profile '@{username}' does not exist on Instagram."
            }
            
        # 2. Fallback to Instaloader if API v1 fails (e.g. rate limit / IP block)
        loader = instaloader.Instaloader(
            download_pictures=False,
            download_videos=False,
            download_video_thumbnails=False,
            download_geotags=False,
            download_comments=False,
            save_metadata=False,
            compress_json=False,
            quiet=True,
        )

        profile = instaloader.Profile.from_username(
            loader.context, username
        )

        fullname = profile.full_name or ""
        bio = profile.biography or ""
        num_followers = profile.followers
        num_following = profile.followees
        num_posts = profile.mediacount
        has_profile_pic = not profile.is_profile_pic_default() if hasattr(profile, 'is_profile_pic_default') else bool(profile.profile_pic_url)
        external_url = bool(profile.external_url)
        is_private = profile.is_private

        return {
            "success": True,
            "username": username,
            "fullname": fullname,
            "bio": bio,
            "num_followers": num_followers,
            "num_following": num_following,
            "num_posts": num_posts,
            "profile_pic": has_profile_pic,
            "external_url": external_url,
            "private": is_private,
            "desc_length": len(bio),
            "profile_pic_url": profile.profile_pic_url,
            "is_verified": profile.is_verified,
        }

    except instaloader.exceptions.ProfileNotExistsException:
        return {
            "success": False,
            "error": f"Instagram blocked our attempt to view '@{username}'. "
                     "Instagram often requires login to view profiles or aggressively blocks AI scrapers. "
                     "Please use the 'Manual Entry' tab instead.",
        }
    except instaloader.exceptions.ConnectionException as e:
        error_msg = str(e).lower()
        if "login" in error_msg or "401" in error_msg:
            return {
                "success": False,
                "error": "Instagram requires login to view this profile. "
                         "The profile may be private or Instagram is "
                         "rate-limiting requests. Please try again later "
                         "or enter profile details manually.",
            }
        return {
            "success": False,
            "error": f"Connection error: {str(e)}. "
                     "Instagram may be rate-limiting. Try again in a few minutes.",
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to fetch profile: {str(e)}. "
                     "You can enter profile details manually instead.",
        }


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    # Test URL parsing
    test_urls = [
        "https://www.instagram.com/meta/",
        "https://instagram.com/python",
        "instagram.com/elonmusk",
        "cristiano",
        "https://www.instagram.com/p/ABC123/",  # Should fail — it's a post
    ]

    print("=" * 50)
    print("  URL PARSING TESTS")
    print("=" * 50)

    for url in test_urls:
        username = extract_username_from_url(url)
        print(f"  {url:45s} → {username or '(invalid)'}")

    print("\n" + "=" * 50)
    print("  PROFILE SCRAPE TEST")
    print("=" * 50)

    result = scrape_instagram_profile("instagram")
    if result["success"]:
        for k, v in result.items():
            if k != "profile_pic_url":
                print(f"  {k:20s}: {v}")
    else:
        print(f"  Error: {result['error']}")
