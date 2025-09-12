
    tax_provider_pairs = {} # Using dict to store {tax_id: provider_name}
    unmatched_providers = list(found_providers)

    for tax in found_taxes:
        best_provider, min_distance = None, float('inf')
        
        for provider in unmatched_providers:
            # Check if they are aligned in the same row or column
            if tax['row'] == provider['row'] or tax['col'] == provider['col']:
                # Calculate distance (number of cells between them)
                distance = abs(tax['row'] - provider['row']) + abs(tax['col'] - provider['col'])
                if distance < min_distance:
                    min_distance, best_provider = distance, provider
        
        if best_provider:
            tax_provider_pairs[tax['text']] = best_provider['text']
            unmatched_providers.remove(best_provider)
        else:
            tax_provider_pairs[tax['text']] = "" # Orphan Tax ID

    # Add any remaining orphan providers
    orphan_providers = [p['text'] for p in unmatched_providers]

    # --- PASS 3: Associate Language and LOBs with the closest Tax ID ---
    all_findings = []
    
    for lang in found_languages:
        closest_tax, min_distance = None, float('inf')
        for tax in found_taxes:
            distance = abs(lang['row'] - tax['row']) + abs(lang['col'] - tax['col'])
            if distance < min_distance:
                min_distance, closest_tax = distance, tax
        
        if closest_tax:
            # Find the closest LOB to this language phrase
            closest_lob, min_lob_dist = "Not Found", float('inf')
            for lob in found_lobs:
                distance = abs(lang['row'] - lob['row']) + abs(lang['col'] - lob['col'])
                # LOB must be between the TIN and the language, or very close
                if distance < min_lob_dist and lob['row'] >= closest_tax['row']:
                     min_lob_dist, closest_lob = distance, lob['text']
            
            all_findings.append({
                "associated_tin": closest_tax['text'],
                "lob_found": closest_lob,
                "provider_name": tax_provider_pairs.get(closest_tax['text'], "Not Found")
            })

    # --- Final Aggregation ---
    lang_present, lang_phrase, all_tins, all_providers, lob_found = "No", "", "", "", ""
    if all_findings:
        lang_present = "Yes"
        lang_phrase = "Phrase Found"
        all_tins = ", ".join(sorted({f["associated_tin"] for f in all_findings if f["associated_tin"]}))
        all_providers = ", ".join(sorted({f["provider_name"] for f in all_findings if f["provider_name"] and f["provider_name"] != "Not Found"}))
        lob_found = ", ".join(sorted({f["lob_found"] for f in all_findings if f["lob_found"] and f["lob_found"] != "Not Found"}))

    return {
        "language_present": lang_present,
        "language_phrase_summary": lang_phrase,
        "associated_tins": all_tins,
        "associated_providers": all_providers,
        "associated_lobs": lob_found,
        "all_providers_found": ", ".join(sorted(list(set(tax_provider_pairs.values())))),
        "all_tins_found": ", ".join(sorted(tax_provider_pairs.keys())),
        "orphan_providers": ", ".join(orphan_providers)
    }
