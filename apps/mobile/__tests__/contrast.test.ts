import { contrast, luminance } from '@/ui/contrast';
import { colors, contrastPairs, type ThemeName } from '@/ui/tokens';

describe('WCAG contrast', () => {
  it('computes known ratios', () => {
    expect(contrast('#000000', '#FFFFFF')).toBeCloseTo(21, 5);
    expect(contrast('#777777', '#FFFFFF')).toBeCloseTo(4.48, 2);
    expect(luminance('#FFFFFF')).toBeCloseTo(1, 6);
    expect(() => luminance('red')).toThrow();
  });

  for (const theme of ['light', 'dark'] as ThemeName[]) {
    it.each(contrastPairs)(`${theme}: $fg on $bg is at least $min:1`, ({ fg, bg, min }) => {
      expect(contrast(colors[theme][fg], colors[theme][bg])).toBeGreaterThanOrEqual(min);
    });
  }
});
