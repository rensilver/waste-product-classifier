from waste_product_classifier import main


def test_main_prints_greeting(capsys):
    main()

    assert "waste-product-classifier" in capsys.readouterr().out
